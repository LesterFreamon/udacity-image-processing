## TODO: define the convolutional neural network architecture

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        self.conv1 = nn.Conv2d(1, 32, 4) # -> (32, 221, 221) -> pooling -> (32, 110, 110)
        self.conv2 = nn.Conv2d(32, 64, 3) # -> (64, 108, 108) -> pooling -> (64, 54, 54)
        self.conv3 = nn.Conv2d(64, 128, 2) # -> (128, 53, 53) -> pooling -> (128, 26, 26)
        self.conv4 = nn.Conv2d(128, 256, 1) # -> (256, 26, 26) -> pooling -> (256, 13, 13)

        self.activation1 = nn.ELU()
        self.max_pooling1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout(p=0.1)
        self.dropout2 = nn.Dropout(p=0.2)
        self.dropout3 = nn.Dropout(p=0.3)
        self.dropout4 = nn.Dropout(p=0.4)
        self.dropout5 = nn.Dropout(p=0.5)
        self.dropout6 = nn.Dropout(p=0.6)


        self.fc1 = nn.Linear(256*13*13, 1000)
        self.fc2 = nn.Linear(1000, 1000)
        self.fc3 = nn.Linear(1000, 136)



        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting

        

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        x = self.dropout1(self.max_pooling1(self.activation1(self.conv1(x))))
        x = self.dropout2(self.max_pooling1(self.activation1(self.conv2(x))))
        x = self.dropout3(self.max_pooling1(self.activation1(self.conv3(x))))
        x = self.dropout4(self.max_pooling1(self.activation1(self.conv4(x))))
        x = x.view(x.size(0), -1)
        x = self.dropout5(self.activation1(self.fc1(x)))
        x = self.dropout6(self.activation1(self.fc2(x)))
        x = self.fc3(x)
        
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
