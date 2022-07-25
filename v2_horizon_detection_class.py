import torch, torchvision
import torch.nn as nn
import numpy as np
import torch
import torch.utils.data
from torch import nn
import torchvision
from torchvision import transforms
import cv2
from networks import VGG_model1 as Net

class detect_horizon():
    def __init__(self):

        self.device = 'cuda:0'
        self.checkpoint = 'saved_models/CNNmodel1.pt'
        self.model_name = 'VGG_model1'
        self.model = model = Net().to(self.device)
        weight = torch.load(self.checkpoint)
        try:
            self.model.load_state_dict(weight['state_dict'])
        except:
            self.model.load_state_dict(weight)
            
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        #
        self.transform = transforms.Compose([
             transforms.ToPILImage(),
             transforms.Resize((224, 224)),
             transforms.ToTensor(),
             normalize,
             self.expand])

    def expand(self, patch):

        patch = torch.unsqueeze(patch, 0)
        return patch.type(torch.float32)

    def detect(self, input):

        input = self.transform(input).to(self.device)
        with torch.no_grad():
            output = self.model(input)

        y_centr = output[0][0] * 1080
        teta = output[0][1] * np.pi/2

        return y_centr, teta