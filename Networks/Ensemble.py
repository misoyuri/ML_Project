import torch
import torchvision            
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms

class MyEnsemble(nn.Module):
    def __init__(self, modelA, modelB, modelC, input):
        super(MyEnsemble, self).__init__()
        self.modelA = modelA
        self.modelB = modelB
        self.modelC = modelC

        self.fc1 = nn.Linear(input, 7)

    def forward(self, x):
        out1 = self.modelA(x)
        out2 = self.modelB(x)
        out3 = self.modelC(x)

        out = out1 + out2 + out3
        out = out / 3
        return F.softmax(out, dim=1)    