from torch import nn

class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        self.flatten = nn.Flatten()
        self.vgg16_stack = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),  
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),         
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),        
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),            
            nn.ReLU(inplace=True),         
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),         
            nn.MaxPool2d(kernel_size=2, stride=2),

            #flattening layer before the linear??
            nn.Flatten(), #testing this out before FC layers
            nn.Linear(25088, 4096), #in should match 512x512 above
            nn.ReLU(inplace=True), #testing this layer out instead of softmax
            nn.Linear(4096,3))#, #3 classes instead of 4, removed Actionable
            #nn.Softmax(dim=1))
        #nn.Sequential(nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)

        #self.linear_layers = Sequential(
        #    Linear(4 * 7 * 7, )
        #)

#transform_test = transforms.Compose([
#    transforms.ToTensor(),
#    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
#])

    def forward(self, x):
        #self.flatten = nn.Flatten()
        #x = self.flatten(x)
        #print('fwd shape x = ',x.shape)
        logits = self.vgg16_stack(x)
        #print('logits out = ', logits.shape)
        
        return logits

model_vgg16 = VGG16() #.to(device)
model_vgg16 = model_vgg16.float()