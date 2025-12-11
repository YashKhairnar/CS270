import yaml
import itertools
from models.cnn import CNNModel
from models.densenet import DenseNetModel

from utils.cachedDataset import CachedImageDataset
from utils.createDataset import VotingDataset, CustomImageDataset
from torch.utils.data import DataLoader, random_split

import torch.nn as nn
from utils.train import train
from utils.test import test, maxVoting
from utils.plotResults import plotResults

import torch
import matplotlib.pyplot as plt
import argparse
import os
from utils.gridSearch import gridSearch

if __name__=='__main__':
    # Take the model type to train
    parser = argparse.ArgumentParser(description='Malware Classification - Grid Search')
    parser.add_argument('--model', type=str, default='cnn', help='which model to train (cnn/densenet)')
    parser.add_argument('--malwareType', type=str, default='f', help='which malware type to train for (a/b/c/d/e/f/all)')
    args = parser.parse_args()
    
    # Open and load config file
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
       
    # load config for model
    dataConfig = config['dataset']

    params = config['params']
    param_grid = {
        "lr": params["learning_rates"],
        "optimizer": params["optimizers"],
        "batch_size": params["batch_sizes"],
        "dropout": params["dropout_rates"],
        "weight_decay": params["weight_decays"],
    }

    # Run grid search
    best_acc, best_params, results = gridSearch(param_grid, dataConfig, args.model, args.malwareType)
    print(f"\n✅ Grid Search Finished!")

    #extract the best parameters
    best_lr = best_params['lr']
    best_optimizer = best_params['optimizer']
    best_batch_size = best_params['batch_size']
    best_dropout = best_params['dropout']
    best_weight_decay = best_params['weight_decay']
    
    #create dataset, dataloader for that train and testing
    # dataset = CachedImageDataset(dataConfig, args.malwareType)
    dataset = CustomImageDataset(dataConfig, args.malwareType)
    n = len(dataset)
    split_index = int(n * 0.8)  # 80:20 split
    train_dataset, test_dataset = random_split(dataset, [split_index, n - split_index])

    train_dataloader = DataLoader(train_dataset, batch_size=best_batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=best_batch_size, shuffle=True)
    
    # For voting: create a separate test dataset that groups images by sample
    if args.malwareType == 'all':
        voting_test_dataset = VotingDataset(dataConfig)

        n_voting = len(voting_test_dataset)
        voting_split = int(n_voting * 0.8)
        _, voting_test_subset = random_split(voting_test_dataset, [voting_split, n_voting - voting_split])

        voting_test_dataloader = DataLoader(voting_test_subset, batch_size=best_batch_size, shuffle=False)


    #create model and setup device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if args.model == 'cnn':
        model = CNNModel().to(device)
        weightspath = '../weights/cnn'

    if args.model == 'densenet':
        if args.malwareType == 'all':
            model = DenseNetModel(num_classes=12).to(device)  # 12 malware families (A-L)
            weightspath = '../weights/densenet_all'
        else:
            model = DenseNetModel(num_classes=12).to(device)
            weightspath = '../weights/densenet'

    
    os.makedirs(weightspath, exist_ok=True)
        
    loss_fn = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=best_lr, weight_decay=best_weight_decay) # only adam for now


    # Lists to store history for plotting
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []

    # for early stopping
    best_acc = 0.0               # best validation/test accuracy seen so far
    patience = 5                 # how many epochs to wait with no improvement
    epochs_no_improve = 0
    best_model_path = f"{weightspath}/{args.model}_{args.malwareType}.pth"  # filename to save best model
    min_delta = 0.0              # minimum change in accuracy to count as improvement

    epochs = 100
    for t in range(epochs):
        print(f"Epoch {t+1}/{epochs}")
        avg_train_loss, train_acc = train(train_dataloader, model, loss_fn, optimizer, device)
        if args.malwareType == 'all':
            avg_test_loss, test_acc = maxVoting(voting_test_dataloader, model, loss_fn, device)
        else:
            avg_test_loss, test_acc = test(test_dataloader, model, loss_fn, device)

        print(
        f"Train loss: {avg_train_loss:.4f}, Train acc: {train_acc:.2f}% | "
        f"Test loss: {avg_test_loss:.4f}, Test acc: {test_acc:.2f}%"
        )

        if test_acc > best_acc + min_delta:
            best_acc = test_acc
            epochs_no_improve = 0
            torch.save({
                "epoch": t,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_acc": best_acc,
            }, best_model_path)

            print(f"New best model saved with test acc = {best_acc:.2f}%")
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve} epoch(s).")

            if epochs_no_improve >= patience:
                print("Early stopping triggered.")
                break

        # Record metrics
        train_losses.append(avg_train_loss)
        train_accuracies.append(train_acc)
        test_losses.append(avg_test_loss)
        test_accuracies.append(test_acc)


    #----------- plot and save the results -----------
    plotResults(train_losses, test_losses, train_accuracies, test_accuracies, args.model, args.malwareType)