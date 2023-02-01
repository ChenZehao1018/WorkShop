import argparse
from utils.helpers import load_pretrain_model,train,predict
import numpy as np
import random
from utils.models import *
from utils.datasets import RoadDataset
def build_network(args):
    """
    Build netowrk according to configuration.
    
    :param args: args dictionary
    
    :returns: desired built network
    """

    if args.model == 'unet':
        model = UNet(n_channels=3, n_classes=2, bilinear=True,
                   dropout=True, cut_last_convblock=False)
    elif args.model=='unetpp':
        """
        model = smp.UnetPlusPlus(
            #encoder_name="resnet34",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            #encoder_name="resnet101",
            encoder_name="resnet101",
            encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
            in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=1,                      # model output channels (number of classes in your dataset)
        ).to(device)
        """
        
        """
        model = smp.MAnet(
            #encoder_name="resnet34",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            #encoder_name="resnet101",
            encoder_name="resnet34",
            encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
            in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=1,                      # model output channels (number of classes in your dataset)
        ).to(device)
        """
        pass
    else:
        raise KeyError("no selected model")
    return model
def build_dataset(args):
    """
    Build dataset according to configuration.
    
    :param args: args dictionary
    
    :returns: desired test dataset
    """

    train_path = args.train_path
    gt_path = args.gt_path
    gt_thresh = args.gt_thresh

    
    dataset = RoadDataset(train_path, gt_path, gt_thresh)
    
    return dataset

def setup_seed(seed):
    """
    Set up random seed to guarantee fixed result
    
    :param seed: random seed
    
    :returns: none
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    # Set random number generator seed for numpy
    parser = argparse.ArgumentParser(description="")

    # optimizer args
    
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-6, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--patience', type=int, default=3, help='patience')
    parser.add_argument('--optim_type', type=str, default='adam', help='choose a optimizer')
    parser.add_argument('--loss_type', type=str, default='cross_entropy', help='choose a loss function')
    
    # model args
    parser.add_argument('--trained_model', action="store_false", help='use previous model parameters')
    parser.add_argument('--model', type=str, default='unet', help='choose a model') # U_Net
    parser.add_argument('--pretrain_weights', type=str, default='vgg13', help='choose a model') # U_Net
    
    # path and constant args
    parser.add_argument('--seed', type=int, default=37, help='random seed')
    parser.add_argument('--train_path', type=str, default='./data/training/aug_images/')
    parser.add_argument('--gt_path', type=str, default='./data/training/aug_groundtruth/')
    parser.add_argument('--checkpoints_path', type=str, default='./checkpoints/')
    parser.add_argument('--gt_thresh', type=float, default=0.5, help='thresh of the groundtruth')
    parser.add_argument('--test_path', type=str, default='./data/test_set_images/')
    parser.add_argument('--output_path', type=str, default='./results/')
    parser.add_argument('--test_model_name', type=str, default='checkpoint_BEST0914.pth')
    parser.add_argument('--predict_patches', action="store_false")
    parser.add_argument('--foreground_threshold', type=float, default=0.25,
                        help='percentage of pixels > 1 required to assign a foreground label to a patch')
    parser.add_argument('--gpu', type=int, default=2, help='gpu index')
    parser.add_argument('--epochs', type=int, default=200, help='epoch')
    parser.add_argument('--split_rate', type=float, default=0.8, help='the proportion of training dataset')
    parser.add_argument('--batch_size', type=int, default=2, help='Number of batch sizes')
    parser.add_argument('--num_workers', type=int, default=2, help='Number of workers when loading data')
    args = parser.parse_args()
    setup_seed(args.seed)
    
    device = torch.device('cuda:'+str(args.gpu) if torch.cuda.is_available() else 'cpu')
    # Build network according to config file and send it to device
    model = build_network(args)
    model.to(device=device)

    # Build dataset according to args file
    dataset = build_dataset(args)
    if args.trained_model:
        
        # Load weights
        checkpoint_path = args.checkpoints_path +args.test_model_name
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))

        # Generate prediction
        predict(args, model, dataset, device)
    else:

        rng = np.random.RandomState(args.seed)
        # Load pretrained models
        if args.pretrain_weights != "no":
            model = load_pretrain_model(model, args)
            print('Loaded pretrained weights!')
        
        train(model, dataset, args, rng=rng, device=device)
