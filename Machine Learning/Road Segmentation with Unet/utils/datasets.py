import os
import glob
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import matplotlib.image as mpimg

#the dataset
class RoadDataset(Dataset):
    def __init__(self, images_dir, gt_dir, gt_thresh):
        self.images_dir = images_dir
        self.gt_dir = gt_dir
        self.resize_test = False
        self.gt_thresh = gt_thresh
        self.ids = [img_name.split('.')[0] for img_name in os.listdir(images_dir)]

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, img, gt_thresh=0.5, is_mask=False, is_test=False):
        img_ndarray = img
        if is_test:
            img_ndarray = np.asarray(img)
            # Normalize test image
            img_ndarray = img_ndarray / 255

        # Set correct channels number and order
        if img_ndarray.ndim == 2 and not is_mask:
            img_ndarray = img_ndarray[np.newaxis, ...]
        elif not is_mask:
            img_ndarray = img_ndarray.transpose((2, 0, 1))

        if not is_test:
            if not is_mask:
                # Transform pixels to integers and normalize train image
                rimg = img_ndarray - np.min(img_ndarray)
                img_ndarray = (rimg / np.max(rimg) * 255).round().astype(np.uint8)
                img_ndarray = img_ndarray / 255
            else:
                # Apply threshold on test image, in order to have categorical labels
                img_ndarray = np.where(img_ndarray > gt_thresh, 1, 0)

        

        return img_ndarray

    def __getitem__(self, idx):
        name = self.ids[idx]

        # Get idx corresponding image and GT
        gt_mask_file = glob.glob(self.gt_dir + name + '.png')
        img_file = glob.glob(self.images_dir + name + '.png')

        # Read the image and mask
        gt_mask = mpimg.imread(gt_mask_file[0])
        img = mpimg.imread(img_file[0])
        raw_mask = gt_mask.copy()

        # Preprocess both image and mask
        img = self.preprocess(img, is_mask=False)
        gt_mask = self.preprocess(gt_mask, gt_thresh=self.gt_thresh,
                                  is_mask=True)

        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'mask': torch.as_tensor(gt_mask.copy()).long().contiguous(),
            'raw_mask': torch.as_tensor(raw_mask.copy()).float().contiguous(),
        }
