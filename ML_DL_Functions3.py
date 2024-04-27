import numpy as np
import torch
import torch.nn as nn
def ID1():
    '''
        Personal ID of the first student.
    '''
    # Insert your ID here
    return 311577589

def ID2():
    '''
        Personal ID of the second student. Fill this only if you were allowed to submit in pairs, Otherwise leave it zeros.
    '''
    # Insert your ID here
    return 000000000

class CNN(nn.Module):
    def __init__(self): # Do NOT change the signature of this function
        super(CNN, self).__init__()
        n = 12
        kernel_size = 3
        padding = 1
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=n,kernel_size=kernel_size,padding=padding)
        self.bn1 = nn.BatchNorm2d(n)
        self.conv2 = nn.Conv2d(in_channels=n, out_channels=2*n, kernel_size=kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm2d(n*2)
        self.conv3 = nn.Conv2d(in_channels=2*n, out_channels=4*n, kernel_size=kernel_size, padding=padding)
        self.bn3 = nn.BatchNorm2d(n*4)
        self.conv4 = nn.Conv2d(in_channels=4*n, out_channels=8*n, kernel_size=kernel_size, padding=padding)
        self.bn4 = nn.BatchNorm2d(n*8)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(8*n * 28 * 14, 100)  
        self.fc2 = nn.Linear(100, 2)
    
    def forward(self,inp):# Do NOT change the signature of this function
        '''
          prerequests:
          parameter inp: the input image, pytorch tensor.
          inp.shape == (N,3,448,224):
            N   := batch size
            3   := RGB channels
            448 := Height
            224 := Width
          
          return output, pytorch tensor
          output.shape == (N,2):
            N := batch size
            2 := same/different pair
        '''
        # out = self.conv1(inp)
        # TODO: complete this function
        n = 12 #
        inp = self.pool(nn.functional.relu(self.conv1(inp)))
        inp = self.bn1(inp)
        inp = self.pool(nn.functional.relu(self.conv2(inp)))
        inp = self.bn2(inp)
        inp = self.pool(nn.functional.relu(self.conv3(inp)))
        inp = self.bn3(inp)
        inp = self.pool(nn.functional.relu(self.conv4(inp)))
        inp = self.bn4(inp)
        inp = inp.reshape(-1, 8 * 28 * 14 * n)  
        inp = nn.functional.relu(self.fc1(inp))
        inp = self.fc2(inp)
        return inp

class CNNChannel(nn.Module):
    def __init__(self):# Do NOT change the signature of this function
        super(CNNChannel, self).__init__()
        # TODO: complete this method
        n = 6  
        kernel_size = 3
        padding = 1
        self.conv1 = nn.Conv2d(in_channels=6, out_channels=n, kernel_size=kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm2d(n)
        self.conv2 = nn.Conv2d(in_channels=n, out_channels=2*n, kernel_size=kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm2d(n*2)
        self.conv3 = nn.Conv2d(in_channels=2*n, out_channels=4*n, kernel_size=kernel_size, padding=padding)
        self.bn3 = nn.BatchNorm2d(n*4)
        self.conv4 = nn.Conv2d(in_channels=4*n, out_channels=8*n, kernel_size=kernel_size, padding=padding)
        self.bn4 = nn.BatchNorm2d(n*8)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(8*n * 14 * 14, 100) 
        self.fc2 = nn.Linear(100, 2)

    # TODO: complete this class
    def forward(self,inp):# Do NOT change the signature of this function
        '''
          prerequests:
          parameter inp: the input image, pytorch tensor
          inp.shape == (N,3,448,224):
            N   := batch size
            3   := RGB channels
            448 := Height
            224 := Width
          
          return output, pytorch tensor
          output.shape == (N,2):
            N := batch size
            2 := same/different pair
        '''
        # TODO start by changing the shape of the input to (N,6,224,224)
        # TODO: complete this function
        n=6
        left_shoes = inp[:, :, :224, :] 
        right_shoes = inp[:, :, 224:, :] 
        inp = torch.cat((left_shoes, right_shoes), dim=1)  
        inp = self.pool(nn.functional.relu(self.conv1(inp)))
        inp = self.bn1(inp)
        inp = self.pool(nn.functional.relu(self.conv2(inp)))
        inp = self.bn2(inp)
        inp = self.pool(nn.functional.relu(self.conv3(inp)))
        inp = self.bn3(inp)
        inp = self.pool(nn.functional.relu(self.conv4(inp)))
        inp = self.bn4(inp)
        inp = inp.reshape(-1, 8 * 14 * 14 * n) 
        inp = nn.functional.relu(self.fc1(inp))
        inp = self.fc2(inp)
        return inp