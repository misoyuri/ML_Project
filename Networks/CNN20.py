import torch
import torchvision            
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms


class CNN20(nn.Module):
    """
        Simple CNN Clssifier
    """
    def __init__(self, num_classes=7):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),nn.LeakyReLU(0.2),
            nn.Conv2d(32, 32, 3, padding=1),nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, 3, padding=1),nn.LeakyReLU(0.2),
            nn.Conv2d(64, 64, 3, padding=1),nn.LeakyReLU(0.2),
            nn.Conv2d(64, 64, 3, padding=1),nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, 3, padding=1),nn.LeakyReLU(0.2),
            nn.Conv2d(128, 128, 3, padding=1),nn.LeakyReLU(0.2),
            nn.Conv2d(128, 128, 3, padding=1),nn.LeakyReLU(0.2),
            nn.Conv2d(128, 128, 3, padding=1),nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, 3, padding=1),nn.LeakyReLU(0.2),
            nn.Conv2d(256, 256, 3, padding=1),nn.LeakyReLU(0.2),
            nn.Conv2d(256, 256, 3, padding=1),nn.LeakyReLU(0.2),
            nn.Conv2d(256, 256, 3, padding=1),nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, 2),

            # nn.Conv2d(512, 512, 3, padding=1),nn.LeakyReLU(0.2),
            # nn.Conv2d(512, 512, 3, padding=1),nn.LeakyReLU(0.2),
            # nn.Conv2d(512, 512, 3, padding=1),nn.LeakyReLU(0.2),
        )
        #512 3 3

        self.avg_pool = nn.AvgPool2d(3)
        #512 1 1
        self.classifier = nn.Sequential(
            nn.Linear(256, 128), 
            nn.Linear(128, 32)   
        )
    
    def forward(self, x):
        features = self.conv(x)
        x = self.avg_pool(features)
        x = x.view(features.size(0), -1)
        x = self.classifier(x)
        x = F.log_softmax(x , dim=1)
        
        return x