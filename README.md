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
 
 Files:
 
 1. get_dbt_data.ipynb
       > This file will open the DBT challenge csv files and map the names and study numbers to view names. From there, the files will convert the dicom images (using dcmread_image) to their array form and save off to a pickle file for later use. There are minor functions used to verify the final outputs. The TRAINING and VALIDATION challenge files were used in this. *NOTE* dcmread_image was embedded into the notebook due to some initial path import problems.
 2. DBT_patch_selection.ipynb  
       > This file is used to generate patches for the Normal, Actionable (extra), Benign, and Cancer cases. It contains all of the functions needed to choose patch areas for normal. The patch selection for each is broken into separate parts for use with Colab (reading the DICOMs and writing patches out takes a lot of time: 2-3 colab sessions for Normal, 1 hour for Benign, <1hr for Cancer patches). A reliable colab session, without idling issues, usually lasts 5 hours, so each has the ability to record the last file written should the user need to pick up from that point due to a colab idle disconnect.


 3. read_single_dbt.ipynb
     > Generic test function used for testing initial setup of get_dbt_data.ipynb (essentially a scratch function)
 4. VGG_Classify.ipynb
     > This is the main function of the VGG Classifier. It contains the building blocks for the VGG16 model format, which includes loading the saved/trained  model and the dataloader configuration. The intention of this code is to test out the final form of VGG classify without RetinaNet inputs. The integrated version of this stored in the google drive is the final version of this. To run, this can use fake box data (to simulate RetinaNet inputs) as inputs or a single box row.
 
 5. VGG_DBT.ipynb
     > This file is the pre-cursor to VGG_Classify, but is not tailored for integration (intended as a stand-alone without RetinaNet expected inputs). It contains most of the same functionality as VGG_Classify. To train, the code is run normally. Once specific criteria are met during the validation run of the model during training, model checkpoints are saved off. Model saves are also done for each epoch. After training, there's a selectable cell that will run the saved model (of choice) and produce various metrics on its performance. The test component of this has a selector to enable it and will run through the saved index of files not used for training (saved during the training dataloader setup) to avoid mixing in trained data. The DBT challenge Test or Validation datasets could be used (in conjuction with faked RetinaNet patch coordinates)
 
 6. VGG_driver.ipynb
       > This is a temporary driver routine to mimic integration and get the classify code ready to be ported to the google drive format/location.

* For the final versions of these, tailored for integration, see the Google Drive DBTdata folder *

Branch: Rina\
  This branch is only for place-holding purposes. Please see the shared Google Drive "DBTdata" and its README for the code.
  

Branch: Joy\
  YOLO pipeline:
  
    - 1_Create_Dateset.ipynb
    - 2_TrainYOLO.ipynb
  Also contains preprocessed dataset [YOLO] (resized 2D images along with their processed true label boxes)  
