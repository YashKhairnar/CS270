import yaml
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from cnn import CNNModel
from utils.createDataset import CustomImageDataset
import numpy as np
from tqdm import tqdm

def load_trained_cnn(model_path, device):
    model = CNNModel()
    checkpoint = torch.load(model_path, map_location=device)

    # check for checkpoint dict or direct state_dict
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        # if checkpoint
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        # if direct state_dict
        model.load_state_dict(checkpoint)

    model.eval()
    model.to(device)
    return model

def extract_features_batch(dataloader, model, device):
    all_features = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Extracting features"):
            images = images.to(device)

            outputs = model(images)

            probabilities = F.softmax(outputs, dim=1)

            all_features.append(probabilities.cpu().numpy())
            all_labels.append(labels.numpy())

    # Concatenate all batches
    features = np.vstack(all_features)
    labels = np.concatenate(all_labels)

    return features, labels


def extract_all_cnn_features(image_types=['a', 'b', 'c', 'd', 'e', 'f']):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    all_sample_features = []
    labels = None

    for img_type in image_types:
        print(f"Processing image type: {img_type}")

        # load config for this image type
        with open('config.yaml', 'r') as file:
            config = yaml.safe_load(file)

        # update config to use this image type
        config['dataset']['malwareSample'] = img_type

        dataset = CustomImageDataset(config['dataset'])
        dataloader = DataLoader(dataset, batch_size=64, shuffle=False)

        model_path = f'weights/CNN/cnn_{img_type}.pth'
        print(f"Loading model: {model_path}")
        model = load_trained_cnn(model_path, device)

        # extract features (N, 12) for this image type
        features, curr_labels = extract_features_batch(dataloader, model, device)
        print(f"Extracted features shape: {features.shape}")

        all_sample_features.append(features)

        # store labels 
        if labels is None:
            labels = curr_labels

    # concatenate features from all 6 models
    # result: (N, 72) where n= No of labels, 72 from 6 models × 12 classes
    X = np.hstack(all_sample_features)

    print(f"Final feature matrix shape: {X.shape}")
    print(f"Labels shape: {labels.shape}")

    return X, labels


if __name__ == '__main__':
    X, y = extract_all_cnn_features()

    np.save('svm_features_X.npy', X)
    np.save('svm_features_y.npy', y)
    print("Features saved to 'svm_features_X.npy' and 'svm_features_y.npy'")