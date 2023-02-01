import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch import optim
from torch.utils.data import DataLoader, Subset
import os
import numpy as np
from PIL import Image
import torch
import os
from tqdm import tqdm
import ttach as tta
import numpy as np
from PIL import Image
from torchvision import transforms
from utils.mask_to_submission import masks_to_submission
import sys


def crop_image(img, w=400, h=400):
    """
    Split image into 4* 400x400 patches

    :param img: Initial image
    :param w: Width of patch
    :param h: Height of patch

    :returns: 4* 400x400 images
    """
    # Get initial image size
    initial_width, initial_height = img.size

    # Setting the points for cropped image
    crop1 = (0, 0, w, h)
    crop2 = (initial_width - w, 0, initial_width, h)
    crop3 = (0, initial_height - h, w, initial_height)
    crop4 = (initial_width - w, initial_height - h, initial_width, initial_height)

    # Cropped image of above dimension
    im1 = img.crop(crop1)
    im2 = img.crop(crop2)
    im3 = img.crop(crop3)
    im4 = img.crop(crop4)

    return [im1, im2, im3, im4]


def overlay_masks(masks, orig_img, mode='avg'):
    """
    Aggregate patches masks into one mask.

    :param masks: Patches of images.
    :param orig_img: Original image.
    :param mode: Way of combining the patches - average or maximum between overlapping areas.

    :returns: one mask.
    """


    # Swap channels order for each masks
    masks = [x.transpose((1, 2, 0)) for x in masks]
    # Get initial test image size
    orig_w, orig_h = orig_img.size
    w, h = masks[0].shape[0], masks[0].shape[1]

    if len(masks) == 1:
        return masks
    else:
        # Build divider image. Each pixel will represent the number of overlapping patches in that point.
        divider = np.ones((orig_w, orig_h))
        divider[(orig_h - h):h, :] = 2
        divider[:, (orig_w - w):w] = 2
        divider[(orig_h - h):h, (orig_w - w):w] = 4
        # Pad patches to same size
        masks[0] = np.pad(masks[0], ((0, orig_h - h), (0, orig_w - w), (0, 0)), 'constant', constant_values=(0, 0))
        masks[1] = np.pad(masks[1], ((0, orig_h - h), (orig_w - w, 0), (0, 0)), 'constant', constant_values=(0, 0))
        masks[2] = np.pad(masks[2], ((orig_h - h, 0), (0, orig_w - w), (0, 0)), 'constant', constant_values=(0, 0))
        masks[3] = np.pad(masks[3], ((orig_h - h, 0), (orig_w - w, 0), (0, 0)), 'constant', constant_values=(0, 0))

        class_masks = []
        for i in range(2):
            mask = np.stack([m[:, :, i] for m in masks], axis=-1)
            if mode == 'avg':
                
                mask = np.sum(mask, axis=-1)
                mask = mask / divider
            else:
                
                mask = np.max(mask, axis=-1)
            
            class_masks.append(mask)
        # Build numpy array
        proba_mask = np.stack(class_masks, axis=-1).transpose((2, 0, 1))

        return proba_mask


def load_pretrain_model(model, args):
    """
    Load pretrained model

    :param model: Network
    :param args: args dictionary

    :returns: a model with loaded parameters
    """
    # Get parameters of VGG13 and own network
    if args.pretrain_weights=="resnet50":
        pass#pretrained_params = resnet50(pretrained=True)
    elif args.pretrain_weights=="vgg13":
        """VGG 13-layer model (configuration "B") with batch normalization"""
        pretrained_params =  torch.load("./pretrained_weights/vgg13_bn-abd245e5.pth")
    elif args.pretrain_weights=="vgg16":
        """VGG 16-layer model (configuration "B") with batch normalization"""
        pretrained_params = torch.load("./pretrained_weights/vgg16_bn-6c64b313.pth")
    
    #print(pretrained_params)
    model_params = model.state_dict()
    mis_match_key={"vgg13":[],"vgg16":["down3.maxpool_conv.2.double_conv.3.weight","down3.maxpool_conv.2.double_conv.0.bias"
    ,"down3.maxpool_conv.2.double_conv.1.weight","down3.maxpool_conv.2.double_conv.1.bias","down3.maxpool_conv.2.double_conv.3.weight","down3.maxpool_conv.2.double_conv.0.weight"]}
    # Load weights according to network configuration
    
    pretrained_keys = list(key for key in pretrained_params.keys() if key.startswith('features'))
    model_keys = list(key for key in model_params.keys() if (key.startswith('inc') or key.startswith('down'))
                            and 'num_batches_tracked' not in key)


   
    # Map own network parameters name to parameters name
    key_mapping = dict(zip(model_keys, pretrained_keys))
    new_state_dict = {}

    # Load params and freeze layers into model
    for model_param_key, model_param_key in model.named_parameters():
        if model_param_key in model_keys and model_param_key not in mis_match_key[args.pretrain_weights]:
            replaced_param = pretrained_params[key_mapping[model_param_key]]
            replaced_param.requires_grad = False
            new_state_dict[model_param_key] = replaced_param
        else:
            new_state_dict[model_param_key] = model_param_key

    # Load named buffers
    for net_buffer_key, net_buffer_value in model.named_buffers():
        new_state_dict[net_buffer_key] = net_buffer_value

    model.load_state_dict(new_state_dict)

    return model


##################### caculate loss ########################

"""
    Focal loss function
    A Focal Loss function addresses class imbalance during training in tasks like object detection. 
    Focal loss applies a modulating term to the cross entropy loss in order to focus learning on hard misclassified examples.
"""
class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)):
            self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, inp, target):
        if inp.dim() > 2:
            inp = inp.view(inp.size(0), inp.size(1), -1)  # N,C,H,W => N,C,H*W
            inp = inp.transpose(1, 2)  # N,C,H*W => N,H*W,C
            inp = inp.contiguous().view(-1, inp.size(2))  # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)
        # Compute logits
        logpt = F.log_softmax(inp)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != inp.data.type():
                self.alpha = self.alpha.type_as(inp.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()

# Average of Dice coefficient for all batches, or for a single mask
def dice_coeff(inp, target, reduce_batch_first=False, epsilon=1e-6):
    assert inp.size() == target.size()
    if inp.dim() == 2 and reduce_batch_first:
        raise ValueError(f'Dice: asked to reduce batch but got tensor without batch dimension (shape {inp.shape})')

    if inp.dim() == 2 or reduce_batch_first:
        inter = torch.dot(inp.reshape(-1), target.reshape(-1))
        sets_sum = torch.sum(inp) + torch.sum(target)
        if sets_sum.item() == 0:
            sets_sum = 2 * inter

        return (2 * inter + epsilon) / (sets_sum + epsilon)
    else:
        # Compute and average metric for each batch element
        dice = 0
        for i in range(inp.shape[0]):
            dice += dice_coeff(inp[i, ...], target[i, ...])
        return dice / inp.shape[0]

# Average of Dice coefficient for all classes
def multiclass_dice_coeff(inp, target, reduce_batch_first=False, epsilon=1e-6):
    assert inp.size() == target.size()
    dice = 0
    for channel in range(inp.shape[1]):
        dice += dice_coeff(inp[:, channel, ...], target[:, channel, ...], reduce_batch_first, epsilon)

    return dice / inp.shape[1]


def dice_loss(inp, target):
    """
    Compute dice loss

    :param inp: Prediction
    :param target: Groundtruth

    :returns: Dice loss
    """
    # Dice loss (objective to minimize) between 0 and 1
    assert inp.size() == target.size()

    return 1 - multiclass_dice_coeff(inp, target, reduce_batch_first=True)

##################### train, validate and test ########################

def train(model, dataset, args, rng, device='cpu'):
    """
    Train the network and save model.

    :param model: built model
    :param dataset: Dataset instance.
    :param args: args dictionary
    :param rng: Random number generator
    :param device: selected device, we should use gpu

    :returns: None
    """
    

    # Split dataset into train and validation subsets
    train_samples = int(len(dataset) * args.split_rate)
    val_samples = len(dataset) - train_samples
    lengths = [train_samples, val_samples]
    indices = rng.permutation(sum(lengths)).tolist()
    train_dataset, val_dataset = [Subset(dataset, indices[offset - length:offset])
                                  for offset, length in zip(np.cumsum(lengths), lengths)]

    # Create dataLoaders
    loader_args = dict(num_workers=args.num_workers, pin_memory=True)
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size, **loader_args)
    val_loader = DataLoader(val_dataset, shuffle=False, batch_size=1, drop_last=True, **loader_args)

    # Set up the optimizer and learning rate scheduler
    if args.optim_type == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum)
    elif args.optim_type == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, momentum=args.momentum)
    elif args.optim_type == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-08,
                               weight_decay=args.weight_decay, amsgrad=False)
    else:
        raise KeyError("no selected optimizer")
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=args.patience)

    # Set loss
    if args.loss_type == 'focal':
        criterion = FocalLoss(gamma=2, alpha=0.25)
    elif args.loss_type == 'cross_entropy':
        criterion = nn.CrossEntropyLoss()
    elif args.loss_type == 'nllloss':
        criterion = nn.NLLLoss()
    else:
        raise KeyError("no selected loss function")

    max_val_score = 0
    print('start training!')
    # Train
    for epoch in tqdm(range(args.epochs)):
        # Train step
        model.train()
        epoch_loss = 0
        for batch in train_loader:
            
            images = batch['image']
            masks = batch['mask']
            images = images.to(device=device, dtype=torch.float32)
            masks = masks.to(device=device, dtype=torch.long)
            
            optimizer.zero_grad()
            pred_masks = model(images)
            
            loss = criterion(pred_masks, masks)
            loss += dice_loss(F.softmax(pred_masks, dim=1).float(), F.one_hot(masks, model.n_classes).
                              permute(0, 3, 1, 2).float())
            ## backward
            loss.backward()
            optimizer.step()

            ## calculate loss
            epoch_loss += loss.item()
        epoch_loss = epoch_loss / len(train_loader)
        
        print('Epoch: {} -> train_loss: {}'.format(epoch,epoch_loss))
        print('start validating')
        # Evaluate model
        val_dice_score, val_loss = evaluate(model, val_loader, epoch, criterion, device)
        
        # Update learning rate
        scheduler.step(val_dice_score)
        
        print('Epoch: {} -> validation_loss: {}'.format(epoch,val_dice_score))
        

        if not os.path.exists(args.checkpoints_path):
            os.makedirs(args.checkpoints_path)
        torch.save(model.state_dict(), args.checkpoints_path + '/checkpoint_' + str(epoch) + '.pth')

        # Save best model if it reaches highest validation score)
        if val_dice_score > max_val_score:
            max_val_score = val_dice_score
            print("Current maximum validation score is: {}".format(max_val_score))
            torch.save(model.state_dict(), args.checkpoints_path + '/checkpoint_BEST{}.pth'.format(max_val_score))


def evaluate(model, dataloader, epoch, criterion, device='cpu'):
    """
    Evaluation

    :param model: built model
    :param dataloader: Validation dataloader.
    :param epoch: current epoch
    :param criterion: Loss function
    :param device: selected device, we should use gpu

    :returns: None
    """

    model.eval()
    # Initialize varibales
    num_val_batches = len(dataloader)
    dice_score = 0
    val_loss = 0
    for i, batch in tqdm(enumerate(dataloader)):
        
        images = batch['image'].to(device=device, dtype=torch.float32)
        masks = batch['mask'].to(device=device, dtype=torch.long)

        binary_mask_one_hot = F.one_hot(masks, model.n_classes).permute(0, 3, 1, 2).float()

        with torch.no_grad():
            # Forward pass
            pred_masks = model(images)
            if model.n_classes == 1:
                # Compute dice score
                pred_masks = (F.sigmoid(pred_masks) > 0.5).float()
                dice_score += dice_coeff(pred_masks, binary_mask_one_hot, reduce_batch_first=False)
            else:
                
                # Compute validation loss
                loss = criterion(pred_masks, masks)
                loss += dice_loss(F.softmax(pred_masks, dim=1).float(), F.one_hot(masks, model.n_classes).
                                  permute(0, 3, 1, 2).float())
                val_loss += loss.item()
                # Compute one hot vectors
                pred_masks = F.one_hot(pred_masks.argmax(dim=1), model.n_classes).permute(0, 3, 1, 2).float()
                # Compute dice score
                dice_score += multiclass_dice_coeff(pred_masks[:, 1:, ...], binary_mask_one_hot[:, 1:, ...],
                                                    reduce_batch_first=False)
    model.train()
    # Update and log validation loss
    val_loss = val_loss / len(dataloader)

    return dice_score / num_val_batches, val_loss


def predict_image(model,initial_img,dataset,device,out_threshold=0.5,test_time_aug=True):
    """
    predict a single patch.

    :param model: built model
    :param initial_img: Entire image or a image patch
    :param dataset: test dataset
    :param device: selected device, we should use gpu
    :param out_threshold: Threshold if a single class is used
    :param test_time_aug: Whether to use test time augmentation or not

    :returns: Probabilities and one-hot mask for initial_img
    """

    model.eval()
    # Pre process test image
    img = torch.from_numpy(dataset.preprocess(initial_img, is_mask=False, is_test=True))
    img = img.unsqueeze(0)
    # Send to device
    img = img.to(device=device, dtype=torch.float32)

    # Define test time augmentations
    aug_images = tta.Compose(
        [tta.HorizontalFlip(), tta.VerticalFlip(), tta.Rotate90(angles=[0,90,180,270])]
    )
    with torch.no_grad():
        if test_time_aug:
            labels = []
            
            for transformer in aug_images:

                augmented_image = transformer.augment_image(img)
                model_output = model(augmented_image)
                deaug_label = transformer.deaugment_mask(model_output)
                labels.append(deaug_label)

            # Reduce results
            bg_mask = torch.cat([t[:, 0, ...] for t in labels], dim=0)
            bg_mask = torch.sum(bg_mask, dim=0) / bg_mask.size(0)
            fg_mask = torch.cat([t[:, 1, ...] for t in labels], dim=0)
            fg_mask = torch.sum(fg_mask, dim=0) / fg_mask.size(0)
            output = torch.cat([bg_mask.unsqueeze(dim=0), fg_mask.unsqueeze(dim=0)], dim=0).unsqueeze(dim=0)
        else:
            # Predict without test time augmentations
            output = model(img)
        # Get probabilities masks
        if model.n_classes > 1:
            probs = F.softmax(output, dim=1)[0]
        else:
            probs = torch.sigmoid(output)[0]

        proba_mask = probs.cpu().squeeze()

        # Get one-hot masks
        if model.n_classes == 1:
            one_hot_mask = (proba_mask > out_threshold).numpy()
        else:
            one_hot_mask = F.one_hot(proba_mask.argmax(dim=0), model.n_classes).permute(2, 0, 1).numpy()

    return proba_mask.numpy(), one_hot_mask


def predict(args, model, dataset, device):

    """
    predict and generate submission

    :param args: args dictionary
    :param model: built model
    :param dataset: test dataset
    :param device: selected device, we should use gpu

    :returns: Probabilities and one-hot mask for initial_img
    """
    # Get test data
    test_folders = os.listdir(args.test_path)
    test_folders = sorted(test_folders, key=lambda x: int(x.split('_')[-1]))

    # Set submission file
    submission_path = args.output_path + "submission_"+args.model +'.csv'
    preds = []
    # Iterate through test images and make predictions
    for i, folder in tqdm(enumerate(test_folders)):
        filename = os.listdir(args.test_path + folder)[0]
        
        img = Image.open(args.test_path + folder + '/' + filename)
        if args.predict_patches:
            # Get 4 * 400x400 patches
            patches = crop_image(img, 400, 400)
        else:
            # Keep entire image
            patches = [img]

        proba_masks = []
        one_hot_masks = []
        # For each sub image, get a prediction
        for patch in patches:
            proba_mask, one_hot_mask = predict_image(model=model,
                                                     initial_img=patch,
                                                     dataset=dataset,
                                                     out_threshold=0.5,
                                                     test_time_aug=True,
                                                     device=device)
            
            proba_masks.append(proba_mask)
            one_hot_masks.append(one_hot_mask)

        if len(proba_masks) > 1:
            # Multiple patches predictions
            proba_mask = overlay_masks(proba_masks, img, mode='avg')
            one_hot_mask = F.one_hot(torch.tensor(proba_mask,
                                                  device='cpu').argmax(dim=0), model.n_classes).permute(2, 0, 1).numpy()
        else:
            # Single image prediction
            proba_mask = proba_masks[0]
            one_hot_mask = one_hot_masks[0]

        # Append foreground mask to predictions
        foreground_mask = proba_mask[1]
        #foreground_mask= morphology.remove_small_objects(foreground_mask.astype(bool), 600)
        preds.append((filename, foreground_mask))


    # Convert mask to submission format and save as csv
    masks_to_submission(submission_path, args.foreground_threshold, preds)
