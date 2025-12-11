import yaml
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from models.cnn import CNNModel
from challenge_dataset import ChallengeDataset

def load_trained_cnn(model_path, device):
    model = CNNModel()
    checkpoint = torch.load(model_path, map_location=device)
    
    # check for checkpoint dict or direct state_dict
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    model.to(device)
    return model

def extract_features_batch(dataloader, model, device):
    all_features = []
    all_sample_ids = []
    
    with torch.no_grad():
        for images, sample_ids in tqdm(dataloader, desc="Extracting features"):
            images = images.to(device)
            
            outputs = model(images)
            probabilities = F.softmax(outputs, dim=1)
            
            all_features.append(probabilities.cpu().numpy())
            all_sample_ids.extend(sample_ids)
    
    # Concatenate all batches
    features = np.vstack(all_features)
    
    return features, all_sample_ids

def extract_challenge_features(image_types=['a', 'b', 'c', 'd', 'e', 'f']):
    device = torch.device('mps')
    
    # Load config
    with open('../src/config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    
    # Point to challenge dataset
    config['dataset']['folderPath'] = '../CS271_final_data/X'
    
    all_sample_features = []
    sample_ids = None
    
    for img_type in image_types:
        print(f"\nProcessing image type: {img_type}")
        
        dataset = ChallengeDataset(config['dataset'], img_type)
        dataloader = DataLoader(dataset, batch_size=64, shuffle=False)
        
        print(f"Found {len(dataset)} samples")
        
        model_path = f'../weights/cnn/cnn_{img_type}.pth'
        print(f"Loading model: {model_path}")
        model = load_trained_cnn(model_path, device)
        
        # extract features (N, 12) for this image type
        features, curr_sample_ids = extract_features_batch(dataloader, model, device)
        print(f"Extracted features shape: {features.shape}")
        
        all_sample_features.append(features)
        
        # store sample IDs from first run
        if sample_ids is None:
            sample_ids = curr_sample_ids
        else:
            # Verify all models process samples in same order
            assert sample_ids == curr_sample_ids, "Sample ordering mismatch across models!"
    
    # concatenate features from all 6 models
    # result: (N, 72) where N = number of samples, 72 from 6 models × 12 classes
    X = np.hstack(all_sample_features)
    
    print(f"\nFinal feature matrix shape: {X.shape}")
    print(f"Number of samples: {len(sample_ids)}")
    
    return X, sample_ids


if __name__ == '__main__':
    print("="*60)
    print("Extracting CNN features from challenge dataset")
    print("="*60)
    
    X, sample_ids = extract_challenge_features()
    
    # Save features and sample IDs
    np.save('challenge_features.npy', X)
    np.save('challenge_sample_ids.npy', np.array(sample_ids))
    
    print(f"\nFeatures saved to 'challenge_features.npy'")
    print(f"Sample IDs saved to 'challenge_sample_ids.npy'")
    print(f"Feature extraction complete!")
