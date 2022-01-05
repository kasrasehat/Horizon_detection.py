from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


class CNNmodel(nn.Module):
    def __init__(self):
        super(CNNmodel, self).__init__()

        self.conv1 = nn.Conv2d(5 , 32, (10, 10), (2, 2), padding = (0, 0)) #(input - kernel_size)/stride +1
        self.conv2 = nn.Conv2d(32, 64, (5, 5)  , (2, 2), padding = (0, 0))
        self.conv3 = nn.Conv2d(64, 128, (3, 3) , (1, 1), padding = (0, 0))
        self.conv4 = nn.Conv2d(128, 32, (1, 1), (1, 1), padding=(0, 0))
        self.dropout1 = nn.Dropout2d(0.33)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(6720, 150)
        self.fc2 = nn.Linear(150, 3)
        self.batchnorm1 = nn.BatchNorm1d(1136)
        self.batchnorm2 = nn.BatchNorm1d(100)
        #self.init_conv2d()

    def init_conv2d(self):
        """
        Initialize convolution parameters using xavier initialization.
        """
        for c in self.children():
            if isinstance(c, nn.Conv2d):
                nn.init.xavier_uniform_(c.weight)
                nn.init.constant_(c.bias, 0.)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        # x = F.max_pool2d(x, 2)
        x = self.conv2(x)
        x = F.relu(x)
        #x = F.max_pool2d(x, 2)
        # x = self.dropout1(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.conv4(x)
        x = torch.flatten(x, 1)
        #X = self.batchnorm1(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        ##X = self.batchnorm2(x)
        x = self.fc2(x)
        return x


class CNNmodel1(nn.Module):
    def __init__(self):
        super(CNNmodel1, self).__init__()

        self.conv1 = nn.Conv2d(5 , 32, (10, 10), (2, 2), padding = (0, 0)) #(input - kernel_size)/stride +1
        self.conv2 = nn.Conv2d(32, 64, (5, 5)  , (2, 2), padding = (0, 0))
        self.conv3 = nn.Conv2d(64, 128, (3, 3) , (1, 1), padding = (0, 0))
        self.conv4 = nn.Conv2d(128, 32, (1, 1), (1, 1), padding=(0, 0))
        self.dropout1 = nn.Dropout2d(0.33)
        self.dropout2 = nn.Dropout2d(0.33)
        self.fc1 = nn.Linear(6720, 150)
        self.fc2 = nn.Linear(150, 1)
        self.fc3 = nn.Linear(6720, 150)
        self.fc4 = nn.Linear(150, 2)
        self.batchnorm1 = nn.BatchNorm1d(1136)
        self.batchnorm2 = nn.BatchNorm1d(100)
        #self.init_conv2d()

    def init_conv2d(self):
        """
        Initialize convolution parameters using xavier initialization.
        """
        for c in self.children():
            if isinstance(c, nn.Conv2d):
                nn.init.xavier_uniform_(c.weight)
                nn.init.constant_(c.bias, 0.)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        # x = F.max_pool2d(x, 2)
        x = self.conv2(x)
        x = F.relu(x)
        #x = F.max_pool2d(x, 2)
        # x = self.dropout1(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.conv4(x)
        x = torch.flatten(x, 1)
        #X = self.batchnorm1(x)
        y = self.fc1(x)
        y = F.relu(y)
        y = self.dropout1(y)
        ##X = self.batchnorm2(x)
        y = self.fc2(y)
        teta =  self.fc3(x)
        teta = F.tanh(teta)
        teta = self.dropout2(teta)
        teta = self.fc4(teta)
        output = [y, teta[:, 0], teta[:, 1]]
        return output