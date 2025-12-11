import os
from torch.utils.data import Dataset
from PIL import Image
import torch
from torchvision import transforms

familyMap = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'J': 9, 'K': 10, 'L': 11}

class CachedImageDataset(Dataset):
    """Dataset that preloads ALL images into RAM for ultra-fast training."""
    
    def __init__(self, dataConfig, malwareType):
        self.malwareSample = malwareType
        self.dataFolder = dataConfig['folderPath']
        self.cached_images = []  # Store preprocessed tensors in RAM
        self.labels = []
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        
        print(f"Loading dataset for malware type '{malwareType}' into RAM...")
        
        # First pass: collect all image paths
        image_paths = []
        for root, _, files in os.walk(self.dataFolder):
            for filename in files:
                if len(filename) == 12:
                    full_path = os.path.join(root, filename)
                    if self.malwareSample != 'all':
                        if filename[6] == self.malwareSample:
                            cls = filename[0]
                            label = familyMap[cls]
                            image_paths.append((full_path, label))
                    else:
                        cls = filename[0]
                        label = familyMap[cls]
                        image_paths.append((full_path, label))
        
        # Second pass: load all images into RAM
        print(f"Caching {len(image_paths)} images into RAM...")
        for idx, (img_path, label) in enumerate(image_paths):
            if idx % 1000 == 0:
                print(f"  Loaded {idx}/{len(image_paths)} images...")
            
            img = Image.open(img_path).convert('RGB')
            img_tensor = self.transform(img)
            
            self.cached_images.append(img_tensor)
            self.labels.append(label)
        
        print(f"✅ All {len(self.cached_images)} images cached in RAM!")
    
    def __len__(self):
        return len(self.cached_images)
    
    def __getitem__(self, idx):
        # Super fast - just return from RAM!
        return self.cached_images[idx], torch.tensor(self.labels[idx], dtype=torch.long)
