import torch
import torch.nn as nn
import torch.nn.functional as F
from models.cnn import CNNModel
from models.densenet import DenseNetModel
from utils.createDataset import CustomImageDataset
from torch.utils.data import DataLoader, random_split
from utils.train import train
from utils.test import test
import itertools
import os
import csv

def gridSearch(param_grid, dataConfig, model_type='cnn', malware_type='f'):
    """
    Perform grid search over hyperparameters.
    
    Args:
        param_grid: Dictionary of parameter lists
        dataConfig: Dataset configuration
        model_type: 'cnn' or 'densenet'
        malware_type: Image type to train on (a-f) or 'all' for voting
    """
    best_acc = 0.0
    best_params = None
    
    # Create all combinations of parameters
    keys, values = zip(*param_grid.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    print(f"\n{'='*80}")
    print(f"Starting Grid Search with {len(combinations)} parameter combinations")
    print(f"Model: {model_type}, Malware Type: {malware_type}")
    print(f"{'='*80}\n")
    
    # Setup device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    
    # Create dataset once
    dataset = CustomImageDataset(dataConfig, malware_type)
    n = len(dataset)
    split_index = int(n * 0.8)
    train_dataset, test_dataset = random_split(dataset, [split_index, n - split_index])
    
    # Results storage
    results = []
    
    # Grid search loop
    for idx, params in enumerate(combinations, 1):
        print(f"\n{'='*80}")
        print(f"Combination {idx}/{len(combinations)}")
        print(f"Parameters: {params}")
        print(f"{'='*80}")
        
        # Create dataloaders with current batch size
        train_dataloader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=params['batch_size'], shuffle=False)
        
        # Create model
        if model_type == 'cnn':
            model = CNNModel().to(device)
        else:  # densenet
            model = DenseNetModel(num_classes=12).to(device)
        
        # Setup optimizer ( only adam for now )
        optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])

        loss_fn = nn.CrossEntropyLoss()
        
        # Training setup
        best_model_acc = 0.0
        patience = 3
        epochs_no_improve = 0
        max_epochs = 5  # Limit epochs for grid search
        
        # Training loop
        for epoch in range(max_epochs):
            avg_train_loss, train_acc = train(train_dataloader, model, loss_fn, optimizer, device)
            avg_test_loss, test_acc = test(test_dataloader, model, loss_fn, device)
            
            if epoch % 5 == 0:  # Print every 5 epochs
                print(f"Epoch {epoch+1}/{max_epochs} - Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%")
            
            if test_acc > best_model_acc:
                best_model_acc = test_acc
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        print(f"\nFinal Test Accuracy: {best_model_acc:.2f}%")
        
        # Save results
        result = {
            'combination': idx,
            'lr': params['lr'],
            'optimizer': params['optimizer'],
            'batch_size': params['batch_size'],
            'dropout': params['dropout'],
            'weight_decay': params['weight_decay'],
            'best_accuracy': best_model_acc,
            'final_train_acc': train_acc,
            'final_test_acc': test_acc
        }
        results.append(result)
        
        # Update global best
        if best_model_acc > best_acc:
            best_acc = best_model_acc
            best_params = params.copy()
    
    print(f"\n{'='*80}")
    print(f"GRID SEARCH COMPLETE")
    print(f"{'='*80}")
    print(f"Best Accuracy: {best_acc:.2f}%")
    print(f"{'='*80}\n")
    
    # Save results to CSV
    os.makedirs('../results', exist_ok=True)
    csv_path = f'../results/gridsearch_{model_type}_{malware_type}.csv'
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    
    print(f"Results saved to: {csv_path}")
    
    return best_acc, best_params, results
