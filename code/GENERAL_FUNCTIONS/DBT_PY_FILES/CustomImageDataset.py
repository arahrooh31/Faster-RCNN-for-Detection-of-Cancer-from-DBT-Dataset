import os
import pandas as pd
from torchvision.io import read_image
from torchvision import transforms  #get normalization functions

class CustomImageDataset(): #Dataset):
    def __init__(self, img_dir,category=[],file_count=1,file_list =[],transform=None, target_transform=None):
        #self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.category = category
        self.file_count = file_count
        self.file_list = file_list
        self.transform = transform
        self.target_transform = target_transform
        self.category_name =''
        



    def image_normalize(image):
        #replace with the more tensor friendly normalize once tensor shapes confirmed
        image = image/65535.0
        return image


    def __len__(self):
        return self.file_count #len(self.file_list) #99 #len(self.img_labels)

    def __getitem__(self, index):
        
        fname = self.file_list[index]

        #get label and pull category 
        text_tokens = fname.split(sep='_')
        label_class = text_tokens[3] #get the label token in 4th position
        self.category_name =  label_class.upper() 


        full_file_name = os.path.join(self.img_dir,self.category_name,fname)
        image = pickle.load( open( full_file_name, "rb" ) )
        image = image.astype(float) #using patch images
        

        #VERIFY THE IMAGES ARE ALL THE SAME 3x244x244
        shapes = image.shape
        assert (shapes[0] == 3),"Image slice error: {0}".format(fname)
        assert (shapes[1] == 244), print('Image row error: ',fname)
        assert (shapes[2] == 244), print('Image column error: ',fname)
        #if (shapes[0] != 3 and shapes[1] != 244 and shapes[2]!= 244):


        #Normalize the data to 0,1 from 2^16
        image = image/65535.0 #image_normalize(image)



        #test out numeric label
        if (label_class in 'Normal'):
            label = 0
        elif (label_class in 'Actionable'):
            print('!!!!! ACTIONABLE PASSED THROUGH')
            stop()
            label = 1
        elif (label_class in 'Benign'):
            label = 1
        else: # (label_class in 'Cancer'):
            label = 2


        #print(full_file_name)


        #read_image(img_path)
        #label = self.img_labels.iloc[idx, 1]
        #if self.transform:
        #    image = self.transform(image)
        #if self.target_transform:
        #    label = self.target_transform(label)
        sample = {"image": image, "label": label}
            #sample = file_name
        return sample

