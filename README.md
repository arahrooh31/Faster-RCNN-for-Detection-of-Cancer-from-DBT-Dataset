# UCLA_BE223C
This repository contains the code for Spring 2021 Bioengineering 223C. 

Contents of repository:



Branch: Main




Branch: Al\
  Contains faster RCNN pipeline\
  Retinanet pipeline that includes evaluation and error analysis


Branch: Keane_temp     
>  This folder contains the pre-processing and VGG model training and testing notebooks before 
        integration. The integrated vgg model exists within the \code directory in the google drive 
        repository. The files and their basic functionality are described below:
 :rocket:
 Files:
 
 1. get_dbt_data.ipynb
       > This file will open the DBT challenge csv files and map the names and study numbers to view names. From there, the files will convert the dicom images (using dcmread_image) to their array form and save off to a pickle file for later use. There are minor functions used to verify the final outputs. The TRAINING and VALIDATION challenge files were used in this. *NOTE* dcmread_image was embedded into the notebook due to some initial path import problems.
 2. DBT_patch_selection.ipynb  
       > This file is used to generate patches for the Normal, Actionable (extra), Benign, and Cancer cases. It contains all of the functions needed to choose patch areas for normal. The patch selection for each is broken into separate parts for use with Colab (reading the DICOMs and writing patches out takes a lot of time: 2-3 colab sessions for Normal, 1 hour for Benign, <1hr for Cancer patches). A reliable colab session, without idling issues, usually lasts 5 hours, so each has the ability to record the last file written should the user need to pick up from that point due to a colab idle disconnect.


 3. read_single_dbt.ipynb
        > Generic test function used for testing initial setup of get_dbt_data.ipynb (essentially a scratch function)
 4. VGG_Classify.ipynb
        > This is the main function of the VGG Classifier. It contains the building blocks for the VGG16 model format, which includes the model and the dataloader configuration to split and use training data for training and validation. In addition to the model, this code has a testing and training setup. To train, the code is run normally. Once specific criteria are met during the validation run of the model during training, model checkpoints are saved off. Model saves are also done for each epoch. After training, there's a selectable cell that will run the saved model (of choice) and produce various metrics on its performance.
 VGG_DBT.ipynb
 VGG_driver.ipynb


Branch: Rina\
  This branch is only for place-holding purposes. Please see the shared Google Drive "DBTdata" and its README for the code.
  

Branch: Joy\
  YOLO pipeline:
  
    - 1_Create_Dateset.ipynb
    - 2_TrainYOLO.ipynb
  Also contains preprocessed dataset [YOLO] (resized 2D images along with their processed true label boxes)  
