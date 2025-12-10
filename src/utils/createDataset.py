import os
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms 
import torch

familyMap = { 'A': 0, 'B':1, 'C':2, 'D':3, 'E':4, 'F':5, 'G':6, 'H':7, 'I':8, 'J':9, 'K':10, 'L':11}

class CustomImageDataset(Dataset):
    def __init__(self, dataConfig, malwareType):
        self.malwareSample = malwareType
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
                    if self.malwareSample !='all': #for cnn and densenet part a)
                        if filename[6]==self.malwareSample:
                            cls = filename[0]
                            label = familyMap[cls]
                            self.imageFiles.append(full_path)
                            self.labels.append(label)
                    else: # for densenet part b) - classify by malware family using all image types
                        cls = filename[0]  # Use malware family (A-L), not image type
                        label = familyMap[cls]  # 12 classes for families A-L
                        self.imageFiles.append(full_path)
                        self.labels.append(label)
    

    def __len__(self):
        return len(self.imageFiles)


    def __getitem__(self,idx):
        image_path = self.imageFiles[idx]
        label = self.labels[idx]

        img = Image.open(image_path).convert('RGB')
        
        # Apply the transforms defined in __init__
        img_tensor = self.transform(img) 

        # Return the image as a Tensor, and the label as a tensor/integer (DataLoader handles label type conversion)
        return img_tensor, torch.tensor(label, dtype=torch.long)
    
    
    def displayImage(self, idx):
        image_path = self.imageFiles[idx]
        label = self.labels[idx]

        img = Image.open(image_path)
        plt.imshow(img)
        plt.title(f"Label: {label} \n Malware type: {self.malwareSample}")
        plt.show()


class VotingDataset(Dataset):
    """Dataset that groups 6 images (types a-f) per sample for voting-based classification."""
    
    def __init__(self, dataConfig):
        self.dataFolder = dataConfig['folderPath']
        self.samples = {}  # Dictionary: sample_id -> {image_type: path, 'label': label}
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        
        # Group images by sample (first 6 chars of filename identify the sample)
        for root, _, files in os.walk(self.dataFolder):
            for filename in files:
                if len(filename) == 12:
                    full_path = os.path.join(root, filename)
                    sample_id = filename[:6]  # First 6 chars identify the sample
                    image_type = filename[6]  # Character at position 6 is the image type (a-f)
                    malware_family = filename[0]  # First char is the malware family (A-L)
                    
                    if sample_id not in self.samples:
                        self.samples[sample_id] = {'label': familyMap[malware_family]}
                    
                    self.samples[sample_id][image_type] = full_path
        
        # Filter to only keep samples that have all 6 image types
        self.valid_samples = []
        for sample_id, data in self.samples.items():
            if all(img_type in data for img_type in ['a', 'b', 'c', 'd', 'e', 'f']):
                self.valid_samples.append(sample_id)
    
    def __len__(self):
        return len(self.valid_samples)
    
    def __getitem__(self, idx):
        sample_id = self.valid_samples[idx]
        sample_data = self.samples[sample_id]
        
        # Load all 6 images for this sample
        images = []
        for img_type in ['a', 'b', 'c', 'd', 'e', 'f']:
            img_path = sample_data[img_type]
            img = Image.open(img_path).convert('RGB')
            img_tensor = self.transform(img)
            images.append(img_tensor)
        
        # Stack images into a single tensor: shape (6, C, H, W)
        images_tensor = torch.stack(images)
        label = sample_data['label']
        
        return images_tensor, torch.tensor(label, dtype=torch.long)