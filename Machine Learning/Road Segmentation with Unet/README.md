# CS-433 Project 1
## Authors (team: mapodoufu)
- Zehao Chen
- Haolong Li
- Xuehan Tang

## Introduction
This repository contains the code for  [Project 2](https://www.aicrowd.com/challenges/epfl-ml-road-segmentation) of Machine Learning 2022 fall at EPFL.

The aim of this project is to train a classifier to segment roads in given images.

## File structure
```shell
│  aug.py
│  README.md
│  requirements.txt
│  run.py
│
├─checkpoints
├─data
├─pretrained_weights
├─results
│      submission_0914.csv
│
└─utils
        datasets.py
        helpers.py
        mask_to_submission.py
        models.py
```
- aug.py : data augmentation python file
- README.md ：simple introduction to this project
- requirements.txt : required python packages in this project
- run.py : main script which train and test the model
- checkpoints/ : store model parameters
- data/ : data folder
- pretrained_weights/ : store pretrained weights
- results/ : the folder to place submission results
- utils/datasets : the python file to generate dataset
- utils/helpers.py : tool functions
- utils/mask_to_submission.py : convert mask to submission
- utils/models.py : the model structure file
## Preparation before predicting

### Prepare the data
Download raw data files from the official [link]([link](https://www.aicrowd.com/challenges/epfl-ml-road-segmentation/dataset_files?unique_download_uri=186486&challenge_id=68))(https://www.aicrowd.com/challenges/epfl-ml-road-segmentation/dataset_files?unique_download_uri=186486&challenge_id=68), unzip training.zip and test_set_images.zip , then put them under data/ folder.

### Download pretrained model 
Since we use pretrained models, to download them, please run following commands from the root folder of the project:
```shell
cd ./pretrained_weights
wget https://download.pytorch.org/models/vgg16_bn-6c64b313.pth
wget https://download.pytorch.org/models/vgg19_bn-c79401a0.pth
cd ..
```

### Download best model
To yield best result, you can download the weights from [link]([link](https://drive.google.com/file/d/1ydE3EUOEjEL1Oc4phRyT-4GgLiLeEEqr/view?usp=sharing))(https://drive.google.com/file/d/1ydE3EUOEjEL1Oc4phRyT-4GgLiLeEEqr/view?usp=sharing)

Put it under ./checkpoints folder.

### Download required python packages
```shell
pip3 install -r requirements.txt
```

## Run the code to generate prediction
From the root folder of the project

```shell
python run.py
```

Then the result will be under the ./results folder.

These were our best results:

| F1-Score    | Accuracy    |
| ----------- | ----------- |
| 0.914       | 0.954        |

## How to train

To train the model

```shell
python run.py --trained_model
```

To change the hyperparameters, you can look into the argparse of run.py to find changeable parameters.