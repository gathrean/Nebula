from torch import nn
from torchsummary import summary
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch

class CNNNetwork(nn.Module):
    
    #constructor for the class
    def __init__(self):
        super().__init__()
        #4 conv blocks / flatten / linear / softmax
        
         # container that has layers, an pytorch will process the layers in order
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1,
                      out_channels=16,
                      kernel_size=3,
                      stride=1,
                      padding=2
            ),
            #rectified linear unit
            nn.ReLU(),
            
            nn.MaxPool2d(kernel_size=2)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16,
                      out_channels=32,
                      kernel_size=3,
                      stride=1,
                      padding=2
            ),
            #rectified linear unit
            nn.ReLU(),
            
            nn.MaxPool2d(kernel_size=2)
        )
                
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32,
                      out_channels=64,
                      kernel_size=3,
                      stride=1,
                      padding=2
            ),
            #rectified linear unit
            nn.ReLU(),
            
            nn.MaxPool2d(kernel_size=2)
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=64,
                      out_channels=128,
                      kernel_size=3,
                      stride=1,
                      padding=2
            ),
            #rectified linear unit
            nn.ReLU(),
            
            nn.MaxPool2d(kernel_size=2)
        )

        #in_features= 2304
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=128,
                      out_channels=256,
                      kernel_size=3,
                      stride=1,
                      padding=2
            ),
            #rectified linear unit
            nn.ReLU(),
            
            nn.MaxPool2d(kernel_size=2)
        )
        
        #in_features= 2048
        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels=256,
                      out_channels=512,
                      kernel_size=3,
                      stride=1,
                      padding=2
            ),
            #rectified linear unit
            nn.ReLU(),
            
            nn.MaxPool2d(kernel_size=2)
        )
        
        #in_features= 4096
        self.conv7 = nn.Sequential(
            nn.Conv2d(in_channels=512,
                      out_channels=1024,
                      kernel_size=3,
                      stride=1,
                      padding=2
            ),
            #rectified linear unit
            nn.ReLU(),
            
            nn.MaxPool2d(kernel_size=2)
        )
        
        #in_features= 8192
        self.conv8 = nn.Sequential(
            nn.Conv2d(in_channels=1024,
                      out_channels=2048,
                      kernel_size=3,
                      stride=1,
                      padding=2
            ),
            #rectified linear unit
            nn.ReLU(),
            
            nn.MaxPool2d(kernel_size=2)
        )
        
        #in_features= 16384
        self.conv9 = nn.Sequential(
            nn.Conv2d(in_channels=2048,
                      out_channels=4096,
                      kernel_size=3,
                      stride=1,
                      padding=2
            ),
            #rectified linear unit
            nn.ReLU(),
            
            nn.MaxPool2d(kernel_size=2)
        )
        
        #flatten the data
        self.flatten = nn.Flatten()
        # out_features is the number of classes (drum, piano, guitar, violin)
        self.linear = nn.Linear(in_features= (128 * 5 * 4), out_features= 13)
        #softmax to normalize the output between the categories
        self.sigmoid = nn.Sigmoid()
        self.dropout_linear = nn.Dropout(0.5)

        #Data augmentation transformations
        self.transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
        ])
    
    def forward(self, input_data):
        # Apply data augmentation
        augmented_input = self.apply_transforms(input_data)
        
        # Resize the input to match the expected size
        augmented_input = F.interpolate(augmented_input, size=(64, 44), mode='bilinear', align_corners=False)

        # pass the results from one layer to the next
        x = self.conv1(augmented_input)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.flatten(x)
        x = x.view(x.size(0), -1)
        x = self.dropout_linear(x)
        predictions = self.linear(x)

        return predictions
    
    def apply_transforms(self, input_data):
        # Apply data augmentation to the input data
        return self.transforms(input_data)
        
if __name__ == "__main__":
    cnn = CNNNetwork()
    
    # Check if CUDA is available
    if torch.cuda.is_available():  # You need to define the not_cpu_heavy_use function
        cnn = cnn.cuda()
        input_size = (1, 64, 44)
        # print the summary of the model, takes model and input size
        summary(cnn, input_size)
        print("CUDA is available")
    else:
        # If CUDA is not available
        input_size = (1, 64, 44)
        # print the summary of the model, takes model and input size
        summary(cnn, input_size)
        print("CUDA not available")