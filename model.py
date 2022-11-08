import torch.nn as nn 
#from torchsummary import summary
import torch 
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, input_size, num_class):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.fc2 = nn.Linear(512, 124)
        self.fc3 = nn.Linear(124, 64)
        self.fc4 = nn.Linear(64, num_class)
    
    def forward(self, xb):
        xb = xb.view(xb.size(0), -1)
        xb = F.relu(self.fc1(xb))
        xb = F.relu(self.fc2(xb))
        xb = F.relu(self.fc3(xb))
        xb = self.fc4(xb)
        out = nn.Softmax(dim=1)(xb)
        return xb

# Lenet model
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)   
        self.conv2 = nn.Conv2d(6, 16, 5) 
        self.fc1 = nn.Linear(16*4*4, 120) 
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
    
    def forward(self, x):
        # reshaping the input tensor to (batch_size, 1, 28, 28)
        x = x.view(x.size(0), 1, 28, 28)
        # print('shape after reshaping', x.shape)
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x