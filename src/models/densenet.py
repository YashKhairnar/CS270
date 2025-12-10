from torchvision import models
import torch.nn as nn
from utils.createDataset import CustomImageDataset
import yaml, torch
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt


class DenseNetModel(nn.Module):
    def __init__(self, num_classes):
        super(DenseNetModel, self).__init__()
        self.model = models.densenet121(weights='DEFAULT')
        #Replace the classifier of the model to give out 12 dim vector for 12 classes
        self.model.classifier = nn.Linear(in_features=1024, out_features=num_classes)
        #freeze all layers except the classifier
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.classifier.parameters():
            param.requires_grad = True
    
    def forward(self, x):
        return self.model(x)