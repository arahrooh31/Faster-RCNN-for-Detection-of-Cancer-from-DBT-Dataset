# -*- coding: utf-8 -*-
"""Faster-RCNN-1.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1wqtUYyAnG1kRXsjs0utCQmeHoWWctA4t
"""

!pip install pydicom
!pip install pickle5
!pip install dicom2nifti
!pip install hiddenlayer

#Dependencies
from torchvision.models.detection import FasterRCNN

import os
import sys
import pandas as pd
from glob import glob
import pickle5 as pickle
import pydicom
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
import numpy as np
from pathlib import Path
import six
import csv
import logging
import math
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import dicom2nifti
import nibabel as nib
from PIL import Image 
import scipy.misc
from scipy import stats
from sklearn.model_selection import train_test_split
from datetime import datetime
from scipy.spatial import distance
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torchvision
from torchvision.models.detection import RetinaNet
from torchvision.models.detection.anchor_utils import AnchorGenerator
import torch.optim as optim
import hiddenlayer as hl

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torchvision
from torchvision.models.detection import RetinaNet
from torchvision.models.detection.anchor_utils import AnchorGenerator
import torch.optim as optim

from torch import nn
import torch.nn.functional as F

from torchvision.ops import MultiScaleRoIAlign

import numpy as np
import cv2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torch.utils.data import DataLoader, Dataset

"""DATALOADER"""

# Mount the google drive so that we can access the data stored in the drive
from google.colab import drive
from google.colab import files
drive.mount('/content/gdrive')

BATCH_SIZE = 6

img_arrays_path = '/content/gdrive/MyDrive/BE223C/DBT_DATA/IMG_ARRAYS/'
boxes_path = '/content/gdrive/MyDrive/BE223C/DBT_DATA/TRAINING_DATA/BCS-DBT boxes-train-v2.csv'
df_boxes = pd.read_csv(boxes_path)
sum(df_boxes['Class'] == 'cancer')

labels_unsplit = []
for i in range(len(df_boxes)):
  if df_boxes['Class'][i] == 'cancer':
    labels_unsplit.append(1)
  else:
    labels_unsplit.append(0)

def select_cases_with_boxes(img_arrays_path, boxes_path):
  df_boxes = pd.read_csv(boxes_path)
  img_names = glob(os.path.join(img_arrays_path, '*.pickle'))
  img_names[0].split('_')[5]
  img_names_valid = []
  for i in range(len(df_boxes)):
    for j in range(len(img_names)):
      img_studyID = img_names[j].split('_')[3]
      view = img_names[j].split('_')[4]
      if df_boxes['StudyUID'][i] == img_studyID and df_boxes['View'][i] == view:
        img_names_valid.append(img_names[j])
  return img_names_valid 

img_names = glob(os.path.join(img_arrays_path, '*.pickle'))

class BreastDataset(Dataset):
  def __init__(self, df_boxes, img_names, batch_size, transform=None):
    self.df_boxes = df_boxes
    self.img_names = img_names
    self.batch_size = batch_size

  def __len__(self):
    return len(self.df_boxes)

  def __getitem__(self, index):
    def padding(array, xx, yy):
    
      h = array.shape[0]
      w = array.shape[1]

      a = (xx - h) // 2
      aa = xx - a - h

      b = (yy - w) // 2
      bb = yy - b - w

      return np.pad(array, pad_width=((a, aa), (b, bb)), mode='constant')
    # Get the image slice where the bounding box resides
    img_array = pickle.load(open(self.img_names[index], "rb" ))
    box_slice_index = self.df_boxes['Slice'][index]
    img_slice = img_array[box_slice_index, :, :]
    # Normalize the image (min max norm)
    img_slice = (img_slice - img_slice.min()) / (img_slice.max() - img_slice.min())
    img_slice = padding(img_slice, 2457, 1996)
    # Convert into a 3 channel image
    img_slice = np.stack((img_slice,)*3, axis = 0)
    img_slice = torch.from_numpy(img_slice.astype(np.float32))
    img_slice = img_slice.to(device = torch.device('cuda:0'))

    # Create the label based on the file name
    if 'Benign' in os.path.basename(self.img_names[index]):
      label = torch.tensor([0])
      label = label.to(device = torch.device('cuda:0'))
    elif 'Cancer' in os.path.basename(self.img_names[index]):
      label = torch.tensor([1])
      label = label.to(device = torch.device('cuda:0'))
    
    # Get the bounding box 
    box = [df_boxes['X'][index], df_boxes['Y'][index], df_boxes['Width'][index], df_boxes['Height'][index]]

    # transform from [x, y, w, h] to [x1, y1, x2, y2]
    box[2] = box[0] + box[2]
    box[3] = box[1] + box[3]
    
    box = torch.as_tensor(box)
    box = box.to(device = torch.device('cuda:0'))
    target = {}
    target["boxes"] = box
    target["labels"] = label

    
    return img_slice, [target] * self.batch_size

len(img_names)
print(img_names[0].split('_')[4])

img_names_valid = select_cases_with_boxes(img_arrays_path, boxes_path)

img_names_valid

breast_dataset = BreastDataset(df_boxes, img_names_valid, BATCH_SIZE)

# For testing the code only. Using 2 samples.
# toy_dataset = BreastDataset(df_boxes.head(10), img_names_valid[0:10])
# toy_dataloader = DataLoader(toy_dataset, batch_size = 2, shuffle = True, num_workers = 0)

# Stratified train, val, test split. First do train-test split, and then take out val from the train
train_indices, test_indices = train_test_split(
    np.arange(len(labels_unsplit)), test_size = 0.1, shuffle = True, stratify = labels_unsplit, random_state = 123)
train_indices, val_indices = train_test_split(
    train_indices, test_size = 0.1, shuffle = True, stratify = [labels_unsplit[i] for i in train_indices], random_state = 123)

train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
val_sampler = torch.utils.data.SubsetRandomSampler(val_indices)
test_sampler = torch.utils.data.SubsetRandomSampler(test_indices)

train_dataloader = DataLoader(breast_dataset, batch_size = BATCH_SIZE, shuffle = False, sampler = train_sampler, num_workers = 0)
val_dataloader = DataLoader(breast_dataset, batch_size = BATCH_SIZE, shuffle = False, sampler = val_sampler, num_workers = 0)
test_dataloader = DataLoader(breast_dataset, batch_size = BATCH_SIZE, shuffle = False, sampler = test_sampler, num_workers = 0)

"""TRAINING THE MODEL"""

def train_val(resume_training, num_epochs):
  num_epochs = num_epochs
  start_epoch = 0  # start from epoch 0 or last epoch

  fasterrcnn = torchvision.models.detection.fasterrcnn_resnet50_fpn(num_classes = 2, pretrained = False, pretrained_backbone = True) 
  fasterrcnn = fasterrcnn.to(device = torch.device('cuda:0'))
  optimizer = optim.SGD(fasterrcnn.parameters(), lr=0.001, momentum=0.9)

  global best_loss
  best_loss = float('inf')  # best test loss
  n_epochs_stop = 10
  epochs_no_improve = 0
  min_delta = 0.5
  early_stop = False

  if resume_training:
    print('==> Resuming from checkpoint..')
    checkpoint = torch.load('/content/gdrive/MyDrive/223C_breast_cancer/code/model_50_epochs_early_stopping_batch_6_momentum6.pth')
    fasterrcnn.load_state_dict(checkpoint['net'])
    best_loss = checkpoint['loss']
    start_epoch = checkpoint['epoch']

  total_epoch_loss_train = []
  total_epoch_loss_val = []
  start_time = datetime.now()
  for epoch in range(start_epoch, start_epoch + num_epochs): 
    # Training phase
    fasterrcnn.train()
    epoch_loss = 0
    epoch_classification_loss = 0
    epoch_regression_loss = 0
    for batch_num, (image, targets) in enumerate(train_dataloader):

      optimizer.zero_grad()
      outputs = fasterrcnn.forward(image, targets)
      classification_loss = outputs['classification']
      regression_loss = outputs['bbox_regression']
      loss = classification_loss + regression_loss 
      loss.backward()
      
      optimizer.step()
      epoch_classification_loss += float(classification_loss)
      epoch_regression_loss += float(epoch_regression_loss)
      epoch_loss += float(loss)

    total_epoch_loss_train.append(epoch_loss/len(train_dataloader))
    print('\n Training')
    print('Epoch: {} | Regression loss: {:1.5f} | Classification loss: {:1.5f} | Epoch loss: {:1.5f}'.format(
        epoch, regression_loss, epoch_classification_loss/(batch_num + 1), epoch_loss/(batch_num + 1)))
        # Validation phase
    epoch_loss_val = 0
    epoch_classification_loss_val = 0
    epoch_regression_loss_val = 0
    for batch_num, (image, targets) in enumerate(val_dataloader):
      outputs_val = fasterrcnn.forward(image, targets)
      classification_loss_val = outputs_val['classification']
      regression_loss_val = outputs_val['bbox_regression']
      loss_val = classification_loss_val + regression_loss_val
      
      epoch_classification_loss_val += float(classification_loss_val)
      epoch_regression_loss_val += float(epoch_regression_loss_val)
      epoch_loss_val += float(loss_val)

    total_epoch_loss_val.append(epoch_loss_val/len(val_dataloader))
    print('\n Validation')
    print('Epoch: {} | Regression loss: {:1.5f} | Classification loss: {:1.5f} | Epoch loss: {:1.5f}'.format(
        epoch, regression_loss_val, epoch_classification_loss_val/(batch_num + 1), epoch_loss_val/(batch_num + 1)))
    
    # # Predictions from the model
    # fasterrcnn.eval()
    # predictions = fasterrcnn.forward(image)
    # print(predictions)
    fasterrcnn.train()
    val_loss = epoch_loss_val/(len(val_dataloader))
    
    # Plot loss for each epoch
    plt.plot(range(epoch+1), total_epoch_loss_train, label = 'Training')
    plt.plot(range(epoch+1), total_epoch_loss_val, label = 'Validation')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # Check if need to do early stopping
    if best_loss - val_loss > min_delta:
      best_loss = val_loss
      epochs_no_improve = 0
    else:
      epochs_no_improve += 1
    if epochs_no_improve == n_epochs_stop:
      print('Early stopping!' )
      early_stop = True
      break
    else:
      print('Save model')
      state = {
        'net': fasterrcnn.state_dict(),
        'loss': val_loss,
        'epoch': epoch,
      }
      torch.save(state, '/content/gdrive/MyDrive/223C_breast_cancer/code/model_50_epochs_early_stopping_batch_6_momentum6.pth')


  # After finishing all the epochs
  end_time = datetime.now()
  print('Duration: {}'.format(end_time - start_time))

train_val(resume_training = False, num_epochs = 50)

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
 
      
def get_object_detection_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    
    num_classes = 2  # 3 class (mark_type_1，mark_type_2) + background

    in_features = model.roi_heads.box_predictor.cls_score.in_features
 
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
 
    return model

!pip install engine

from engine import train_one_epoch, evaluate
import utils


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

num_classes = 2

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False, progress=True, num_classes=num_classes, pretrained_backbone=True)  # Or get_object_detection_model(num_classes)
model.to(device)

params = [p for p in model.parameters() if p.requires_grad]

optimizer = torch.optim.SGD(params, lr=0.0003, momentum=0.9, weight_decay=0.0005)

lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=1, T_mult=2)


num_epochs = 31

for epoch in range(num_epochs):
    # train for one epoch, printing every 10 iterations
    # Engine.pyTrain_ofOne_The epoch function takes both images and targets. to(device)
    train_one_epoch(model, optimizer, train_dataloader, device, epoch, print_freq=50)

    # update the learning rate
    lr_scheduler.step()

    # evaluate on the test dataset    
    evaluate(model, data_loader_test, device=device)    
    
    print('')
    print('==================================================')
    print('')

print("That's it!")

!pip install pycocotools

import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
 

# Load the pre-trained pre-trained model on COCO
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
#This operation you really need to have fixed parameters
for param in model.parameters():
    param.requires_grad = False
    
# Replace the classifier with a new classifier with user-defined num_classes
num_classes = 2  # 1 class (person) + background

# Get the number of input parameters of the classifier
in_features = model.roi_heads.box_predictor.cls_score.in_features
# Replace the pre-trained head with a new head
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)


# Load the pre-trained model for classification and return only features
backbone = torchvision.models.mobilenet_v2(pretrained=True).features
# FasterRCNN needs to know the number of output channels in the backbone network. For mobilenet_v2, it is 1280, so we need to add it here
backbone.out_channels = 1280
 
# We let RPN generate 5 x 3 Anchors (with 5 different sizes and 3 different aspect ratios) at each spatial position
# We have a tuple [tuple [int]], because each feature map may have a different size and aspect ratio
anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                   aspect_ratios=((0.5, 1.0, 2.0),))
 
# Define the feature map that we will use to perform the cropping of the region of interest, and the size of the crop after rescaling.
# If your trunk returns Tensor, featmap_names should be [0].
# More generally, the trunk should return OrderedDict [Tensor]
# And in featmap_names, you can choose the feature map you want to use.
roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=[0],#featmap_names=['0']
                                                output_size=7,
                                                sampling_ratio=2)
# Put these pieces in the FasterRCNN model
model = FasterRCNN(backbone,
                   num_classes=2,
                   rpn_anchor_generator=anchor_generator,
                   box_roi_pool=roi_pooler)

import pathlib

import albumentations as A
import numpy as np
from torch.utils.data import DataLoader

from datasets import ObjectDetectionDataSet
from transformations import ComposeDouble, Clip, AlbumentationWrapper, FunctionWrapperDouble
from transformations import normalize_01
from utils import get_filenames_of_path, collate_double

params = {'BATCH_SIZE': 2,
          'LR': 0.001,
          'PRECISION': 32,
          'CLASSES': 2,
          'SEED': 42,
          'PROJECT': 'Heads',
          'EXPERIMENT': 'heads',
          'MAXEPOCHS': 500,
          'BACKBONE': 'resnet34',
          'FPN': False,
          'ANCHOR_SIZE': ((32, 64, 128, 256, 512),),
          'ASPECT_RATIOS': ((0.5, 1.0, 2.0),),
          'MIN_SIZE': 1024,
          'MAX_SIZE': 1024,
          'IMG_MEAN': [0.485, 0.456, 0.406],
          'IMG_STD': [0.229, 0.224, 0.225],
          'IOU_THRESHOLD': 0.5
          }


from faster_RCNN import get_fasterRCNN_resnet

model = get_fasterRCNN_resnet(num_classes=params['CLASSES'],
                              backbone_name=params['BACKBONE'],
                              anchor_size=params['ANCHOR_SIZE'],
                              aspect_ratios=params['ASPECT_RATIOS'],
                              fpn=params['FPN'],
                              min_size=params['MIN_SIZE'],
                              max_size=params['MAX_SIZE'])

from faster_RCNN import FasterRCNN_lightning

task = FasterRCNN_lightning(model=model, lr=params['LR'], iou_threshold=params['IOU_THRESHOLD'])

from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping

checkpoint_callback = ModelCheckpoint(monitor='Validation_mAP', mode='max')
learningrate_callback = LearningRateMonitor(logging_interval='step', log_momentum=False)
early_stopping_callback = EarlyStopping(monitor='Validation_mAP', patience=50, mode='max')

# trainer init
from pytorch_lightning import Trainer

trainer = Trainer(gpus=1,
                  precision=params['PRECISION'],  # try 16 with enable_pl_optimizer=False
                  callbacks=[checkpoint_callback, learningrate_callback, early_stopping_callback],
                  default_root_dir='heads',  # where checkpoints are saved to
                  logger=neptune_logger,
                  log_every_n_steps=1,
                  num_sanity_val_steps=0,
                  enable_pl_optimizer=False,  # False seems to be necessary for half precision
                  )