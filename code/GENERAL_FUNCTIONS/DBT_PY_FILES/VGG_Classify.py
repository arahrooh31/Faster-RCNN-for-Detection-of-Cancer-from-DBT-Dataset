####################################################
# MAIN CLASSIFICATION CODE
####################################################

def VGG_Classify(img_data, box_info,model_file):
    #img_data is the full dicom volume a 3D array
    #box_info contains the coordinates to one RetinaNet annotation
    #model_file is the saved checkpoint of the VGG16 model

    import torch
    import numpy as np
    from torch.utils.data import DataLoader
    from torch import FloatTensor
    from torch import tensor

    from TestImageDataset import TestImageDataset
    from randomize_patches import randomize_patches
    
    ### Enable GPU, if present
    train_on_gpu = torch.cuda.is_available()
    if (train_on_gpu):
        dev=torch.device("cuda")

    from VGG16 import VGG16

    if (train_on_gpu):
        model_vgg16 = VGG16().to(dev)
    else:
        model_vgg16 = VGG16() #.to(device)
    model_vgg16 = model_vgg16.float()
    
    #keep all of the model outputs for a final result
    stored_predictions = [] #keep all of the predictions 
    stored_probabilities =[] #keep all of the softmax values for plots
    
    #check input image
    islice, irows, icols = np.shape(img_data)

    ############################################################################
    # LOAD MODEL FROM ARGUMENTS
    if (train_on_gpu):
        checkpoint = torch.load(model_file,map_location=torch.device('cuda'))
    else:
        checkpoint = torch.load(model_file, map_location=torch.device('cpu'))

    model_vgg16.load_state_dict(checkpoint)
    model_vgg16.eval()


    ############################################################################
    # Parse through argument box info
    ############################################################################
    start_point  = -1
    patch_x = 244
    patch_y = 244

    slice_selection = box_info[1]
    x_corner = box_info[2]
    y_corner = box_info[3]
    x_width = box_info[4]
    y_height = box_info[5]

    #for one retinanet box, generate X number of randomized boxes about that corner
    row_lims_low, row_lims_high, col_lims_low, col_lims_high = randomize_patches(islice, irows, icols,slice_selection, x_corner, y_corner, x_width,y_height)

    slice_lower = slice_selection - 1
    slice_upper = slice_selection + 1


    #build patches from random boxes
    for ii in range(0,len(row_lims_low)):

        patch_img = img_data[slice_lower:slice_upper+1,row_lims_low[ii]:row_lims_high[ii],col_lims_low[ii]:col_lims_high[ii]]

        #Do Test Portion here
        #1. Take in random patch, use transforms to make rotations and a flip
        #2. Return those as the batch images
        #3. Run Model over that set
        #4. Take any that match cancer as the label

        #load up with the pre-sized patch images
        all_data = TestImageDataset(patch_data = patch_img,
                                    file_count=1, #full_file_count,
                                    transform=None, 
                                    target_transform=None)

        #del patch_img
        
        dataloader_all = DataLoader(all_data, batch_size=1,shuffle=True, num_workers=2)#, 

        for epoch in range(0,1):
            with torch.no_grad():
                for i, data in enumerate(dataloader_all, 0):
    
                    num_images = len(data['image'])
                    for ii in range(0,num_images):
    
                        inputs = data['image'][ii].type(FloatTensor)
                        
                        #if (train_on_gpu):
                        inputs = inputs.cuda()
    
                        # forward + backward + optimize
                        outputs = model_vgg16(inputs) #.permute(0, 1, 2, 3))
                        #print('Model outputs complete') #debug
                        outputs=torch.flatten(outputs, start_dim=1)
                        #loss = criterion(outputs, labels.long())
    
    
                        #print(outputs)
                        y_pred_softmax = torch.log_softmax(outputs, dim = 1)
                        _, y_pred_tags = torch.max(y_pred_softmax, dim = 1)
                        #print('Predicted Value = ',ii,y_pred_tags)
                        stored_predictions.append(y_pred_tags)
                        stored_probabilities.append(y_pred_softmax)

    
    #memory cleanup
    del dataloader_all
    del inputs
    del outputs
    
    #build totals of cancer and non-cancer
    cancer = 0
    no_cancer = 0
    cancer_cutoff = 0.5 #results > x % are labeled cancer
    final_prediction = 0 #default is no cancer label
    for ii in range(0,len(stored_predictions)):
        if (stored_predictions[ii] == 2):
            cancer+=1
        else: #0 and 1 labels lumped together
            no_cancer +=1
    if (cancer >= (cancer_cutoff * len(stored_predictions)) ):
        final_prediction = 1 #change this to cancer and output
    
    return final_prediction, stored_probabilities


