import os
import argparse

import cv2
import imutils
import numpy as np
import random
def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
setup_seed(37)
def rotate(img, gt,angle):
    new_image=imutils.rotate(img, angle)
    new_gt=imutils.rotate(gt, angle)
    return new_image, new_gt

def flip(img, gt,flip_type):
    new_image=cv2.flip(img, flip_type)
    new_gt=cv2.flip(gt, flip_type)
    return new_image, new_gt

def add_contrast_and_brightness(img, gt):
    min_contrast, max_contrast = 0.7, 1.3
    min_bright, max_bright = -70, 70
    new_image = np.clip(img * np.random.uniform(min_contrast, max_contrast) +
                        np.random.uniform(min_bright, max_bright), 0, 255)
    return new_image,gt

def add_gamma(img, gt,gamma):
    inv_gamma = 1.0 / gamma
    table = np.array([((k / 255.0) ** inv_gamma) * 255 for k in np.arange(0, 256)]).astype("uint8")
    new_image = cv2.LUT(img, table)
    return new_image,gt

def sharpen(img, gt):
        
    kernel = np.array([[0, -1, 0],
                    [-1, 5, -1],
                    [0, -1, 0]])

    # apply kernel
    sharpened_image = cv2.filter2D(img, -1, kernel)

    return sharpened_image,gt

if __name__ == '__main__':
    # Get command line arguments and configuration dictionary
    
    parser = argparse.ArgumentParser(description="")
    # optimizer args

    parser.add_argument('--train_data_path', type=str, default="data/training/images/")
    parser.add_argument('--train_gt_path', type=str, default="data/training/groundtruth/")
    parser.add_argument('--train_aug_data_path', type=str, default="data/training/aug_images/")
    parser.add_argument('--train_aug_gt_path', type=str, default="data/training/aug_groundtruth/")
    args = parser.parse_args()
    # Get all basic training images
    if not os.path.exists(args.train_aug_data_path):
        os.mkdir(args.train_aug_data_path)
    if not os.path.exists(args.train_aug_gt_path):
        os.mkdir(args.train_aug_gt_path)
    
    train_images_names = os.listdir(args.train_data_path)
    

    for i, img_filename in enumerate(train_images_names):
        #copy origin images

        multi_index=0
        os.system("cp {} {}".format(args.train_data_path + img_filename,args.train_aug_data_path+"satImage_" + str(multi_index*100+i+1).zfill(4) + ".png"))
        os.system("cp {} {}".format(args.train_gt_path + img_filename,args.train_aug_gt_path+"satImage_" + str(multi_index*100+i+1).zfill(4) + ".png"))
        multi_index+=1  

        #rotate images 
        for angle in [45, 90, 135, 180, 225, 270, 315]:
            img = cv2.imread(args.train_data_path + img_filename)
            gt = cv2.imread(args.train_gt_path + img_filename, cv2.IMREAD_GRAYSCALE)
            new_image, new_gt=rotate(img, gt,angle)
            img_name = args.train_aug_data_path+"satImage_" + str(multi_index*100+i+1).zfill(4) + ".png"
            gt_name =args.train_aug_gt_path+"satImage_" + str(multi_index*100+i+1).zfill(4) + ".png"
            cv2.imwrite(img_name, new_image)
            cv2.imwrite(gt_name, gt)
            multi_index+=1
        
        #flip images 
        for flip_type in [0, 1]:
            img = cv2.imread(args.train_data_path + img_filename)
            gt = cv2.imread(args.train_gt_path + img_filename, cv2.IMREAD_GRAYSCALE)
            new_image, new_gt=flip(img, gt,flip_type)
            img_name = args.train_aug_data_path+"satImage_" + str(multi_index*100+i+1).zfill(4) + ".png"
            gt_name =args.train_aug_gt_path+"satImage_" + str(multi_index*100+i+1).zfill(4) + ".png"
            cv2.imwrite(img_name, new_image)
            cv2.imwrite(gt_name, gt)
            multi_index+=1
        
        #add_contrast_and_brightness
        img = cv2.imread(args.train_data_path + img_filename)
        gt = cv2.imread(args.train_gt_path + img_filename, cv2.IMREAD_GRAYSCALE)
        new_image, new_gt=add_contrast_and_brightness(img, gt)
        img_name = args.train_aug_data_path+"satImage_" + str(multi_index*100+i+1).zfill(4) + ".png"
        gt_name =args.train_aug_gt_path+"satImage_" + str(multi_index*100+i+1).zfill(4) + ".png"
        cv2.imwrite(img_name, new_image)
        cv2.imwrite(gt_name, gt)
        multi_index+=1

        #sharpen image
        img = cv2.imread(args.train_data_path + img_filename)
        gt = cv2.imread(args.train_gt_path + img_filename, cv2.IMREAD_GRAYSCALE)
        new_image, new_gt=sharpen(img, gt)
        img_name = args.train_aug_data_path+"satImage_" + str(multi_index*100+i+1).zfill(4) + ".png"
        gt_name =args.train_aug_gt_path+"satImage_" + str(multi_index*100+i+1).zfill(4) + ".png"
        cv2.imwrite(img_name, new_image)
        cv2.imwrite(gt_name, gt)
        multi_index+=1
        
        #add gamma
        for gamma in [0.4,0.7, 1.0, 1.3]:
            img = cv2.imread(args.train_data_path + img_filename)
            gt = cv2.imread(args.train_gt_path + img_filename, cv2.IMREAD_GRAYSCALE)
            new_image, new_gt=add_gamma(img, gt,gamma)
            img_name = args.train_aug_data_path+"satImage_" + str(multi_index*100+i+1).zfill(4) + ".png"
            gt_name =args.train_aug_gt_path+"satImage_" + str(multi_index*100+i+1).zfill(4) + ".png"
            cv2.imwrite(img_name, new_image)
            cv2.imwrite(gt_name, gt)
            multi_index+=1
        
        print("image{} has been converted".format(i))
        """"""