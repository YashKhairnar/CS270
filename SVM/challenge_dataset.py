import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

class ChallengeDataset(Dataset):
    """Dataset for loading challenge images without labels for inference."""
    
    def __init__(self, dataConfig, image_type):
        """
        Args:
            dataConfig: Dictionary with 'folderPath' key pointing to challenge data folder
            image_type: One of 'a', 'b', 'c', 'd', 'e', 'f'
        """
        self.dataFolder = dataConfig['folderPath']
        self.imageType = image_type
        self.samples = []
        
        # Define transforms: same as training
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        
        # Collect all samples for this image type
        # Files are named like: X_00000_a.tiff, X_00001_a.tiff, etc.
        for filename in os.listdir(self.dataFolder):
            if filename.endswith(f'_{image_type}.tiff'):
                # Extract sample ID from filename
                # Example: X_00123_a.tiff -> sample_id = 00123
                parts = filename.split('_')
                if len(parts) == 3 and parts[0] == 'X':
                    sample_id = parts[1]
                    full_path = os.path.join(self.dataFolder, filename)
                    self.samples.append((sample_id, full_path))
        
        # Sort by sample ID to ensure consistent ordering
        self.samples.sort(key=lambda x: x[0])
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample_id, image_path = self.samples[idx]
        
        # Load and transform image
        img = Image.open(image_path).convert('RGB')
        img_tensor = self.transform(img)
        
        # Return image tensor and sample ID (no label for inference)
        return img_tensor, sample_id
