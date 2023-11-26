from torch import nn
from torchsummary import summary
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
        
        #flatten the data
        self.flatten = nn.Flatten()
        # out_features is the number of classes (drum, piano, guitar, violin)
        self.linear = nn.Linear(in_features= (128 * 5 * 4), out_features= 11)
        #softmax to normalize the output between the categories
        self.sigmoid = nn.Sigmoid()
    
    
    def forward(self, input_data):
        # pass the results from one layer to the next
        x = self.conv1(input_data)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.flatten(x)
        predictions = self.linear(x)
        
        return predictions
        
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