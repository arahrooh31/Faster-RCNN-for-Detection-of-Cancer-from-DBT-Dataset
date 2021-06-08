import torch
import os
import cv2
from PIL import Image
import numpy as np
from torchvision import transforms
from torch.autograd import Variable
import torch.nn.functional as F

from VGG16 import VGG16

from torchvision.utils import make_grid, save_image
from torchsummary import summary

IMAGE_HEIGHT = 244 
IMAGE_WIDTH = 244  
IMG_FILE = os.path.join('images', 'Index_2_center_slice_1.png')
model_file_v1 = '/content/gdrive/Shareddrives/DBTdata/code/VGG_MODELS/vgg16_best_accuracy_93_EPOCH_96_0.04582521319389343'
model_file_v2 = '/content/gdrive/Shareddrives/DBTdata/code/VGG_MODELS/vgg16_best_accuracy_99_EPOCH_94_0.00040225533302873373'

USE_V2 = False

#hook into layer in model
class SaveFeatures(): 
    features=None
    def __init__(self, m): self.hook = m.register_forward_hook(self.hook_fn)  # attach the hook to the specified layer
    def hook_fn(self, module, input, output): self.features = ((output.cpu()).data).numpy() # copy the activation features as an instance variable
    def remove(self): self.hook.remove()
    
#generate heatmap from features and overlay
def visualize_cam(mask, img, alpha):
    
    mask = cv2.resize(mask, dsize=(IMAGE_HEIGHT, IMAGE_WIDTH), interpolation=cv2.INTER_CUBIC)

    max_mask = np.amax(mask)
    heatmap = (255 * mask/max_mask).astype(np.uint8)

    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = torch.from_numpy(heatmap).permute(2, 0, 1).float().div(255)
    b, g, r = heatmap.split(1)
    heatmap = torch.cat([r, g, b]) * alpha

    result = heatmap+img.cpu()
    result = result.div(result.max()).squeeze()

    return heatmap, result

def main(IMG_FILE, save_path):

    if USE_V2:
        model_file = model_file_v2
        out_file = 'Index_2_center_slice_1_gradcam_v2.png'
    else:
        model_file = model_file_v1
        out_file = 'Index_2_center_slice_1_gradcam_v1.png'
    
    transformation = transforms.Compose([
        transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
        transforms.ToTensor()
    ])
    
    #open, load, convert image to tensor
    img = Image.open(IMG_FILE).convert("RGB")
    img = transformation(img)
    img = img.unsqueeze(0) #add dimension so single image can be run through model

    
    if torch.cuda.is_available():
        device = "cuda"
        img = img.cuda()
    else:
        device = "cpu"
     
    checkpoint = torch.load(model_file)
    model = VGG16()
    model.load_state_dict(checkpoint)
    model.eval().to(device)
    
    #print model structure
    #print(model)
    #summary(model, (3,244,244))

    final_layer = model.vgg16_stack[22] #last maxpool2d layer obtained from print(model) and summary(model)
    activated_features = SaveFeatures(final_layer)
    prediction_var = Variable(img, requires_grad=True)
    
    model(prediction_var) #run the sample through the model to actually get parameters
    
    #reduce maxpool2d parameters from (512, 7, 7) to (1, 7, 7) heatmap by averaging values at each "pixel"
    mask = np.average(activated_features.features, axis = 1) 
    
    #generate heatmap from upscaled mask, then output all images onto a grid and save
    #top = original, middle = heatmap only, bottom = original with heatmap
    images = []
    heatmap, result = visualize_cam(mask[0], img, 0.6)
    images.extend([img.cpu().squeeze(), heatmap, result])
    
    grid_image = make_grid(images, nrow=1)
    gradcam_img = transforms.ToPILImage()(grid_image)
    gradcam_img.save(os.path.join(save_path, out_file))
    print("Done!")
    
    
if __name__ == "__main__":
    main()