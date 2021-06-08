from create_augmented_data import create_augmented_data
class TestImageDataset(): 
    def __init__(self, patch_data,file_count=1,transform=None, target_transform=None):
        #self.img_labels = pd.read_csv(annotations_file)
        self.patch_data = patch_data

        self.file_count = file_count

        self.transform = transform
        self.target_transform = target_transform

        



    def image_normalize(self,image):
        #replace with the more tensor friendly normalize once tensor shapes confirmed
        image = image/65535.0
        return image


    def __len__(self):
        return self.file_count #len(self.file_list) #99 #len(self.img_labels)

    def __getitem__(self, index):

        image = self.patch_data
        image.astype(float)

        #fname = self.file_list[index]

        #VERIFY THE IMAGES ARE ALL THE SAME 3x244x244
        shapes = image.shape
        assert (shapes[0] == 3),   print('Image slice error: ',shapes[0])
        assert (shapes[1] == 244), print('Image row error: ',shapes[1])
        assert (shapes[2] == 244), print('Image column error: ',shapes[2])
        #if (shapes[0] != 3 and shapes[1] != 244 and shapes[2]!= 244):


        #Normalize the data to 0,1 from 2^16
        image = self.image_normalize(image) #image/65535.0 #image_normalize(image)


        output90, output180, output270, outputflip = create_augmented_data(image,flip = 1,rot90=1,rot180=1,rot270=1)


        out_data = [output90,output180, output270, outputflip]


        sample = {"image": out_data}
        return sample