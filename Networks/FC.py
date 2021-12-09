import torch
import torchvision            
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms

class SimpleFC(nn.Module):
    """
        Simple CNN Clssifier
    """
    def __init__(self, num_classes=7):
        super(SimpleFC, self).__init__()

        #1 48 48
        self.classifier = nn.Sequential(
            nn.Linear(48*48, 128), 
            nn.Linear(128, 7)   
        )
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        x = F.log_softmax(x , dim=1)
        
        return x