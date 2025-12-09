import os
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch
# Import transforms for data manipulation
from torchvision import transforms 

class CustomImageDataset(Dataset):
    def __init__(self, dataConfig):
        self.malwareSample = dataConfig['malwareSample']
        self.dataFolder = dataConfig['folderPath']
        self.imageFiles = []
        self.labels = []

        # Define transforms: Resize images and convert them to a PyTorch Tensor
        # The ToTensor transform also handles the necessary (H, W, C) -> (C, H, W) dimension swap
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)), # Common size for CNNs (adjust as needed)
            transforms.ToTensor(),         # Converts PIL image to FloatTensor and scales pixel values to [0, 1]
        ])

        for root, _ , files in os.walk(self.dataFolder):
            for filename in files:
                if len(filename) == 12:
                    full_path = os.path.join(root, filename)
                    self.imageFiles.append(full_path)
                    if filename[6]==self.malwareSample:
                        self.labels.append(1)
                    else:
                        self.labels.append(0)
    

    def __len__(self):
        return len(self.imageFiles)

    def __getitem__(self,idx):
        image_path = self.imageFiles[idx]
        label = self.labels[idx]

        img = Image.open(image_path).convert('RGB')
        
        # Apply the transforms defined in __init__
        img_tensor = self.transform(img) 

        # Return the image as a Tensor, and the label as a tensor/integer (DataLoader handles label type conversion)
        return img_tensor, label
    
    def displayImage(self, idx):
        image_path = self.imageFiles[idx]
        label = self.labels[idx]

        img = Image.open(image_path)
        plt.imshow(img)
        plt.title(f"Label: {label} \n Malware type: {self.malwareSample}")
        plt.show()