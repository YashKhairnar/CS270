import yaml
from cnn import CNNModel
from createDataset import CustomImageDataset
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
from train import train
from test import test
import torch
import matplotlib.pyplot as plt

if __name__=='__main__':
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
       
    # load config for cnn model
    dataConfig = config['dataset']

    #create dataset, dataloader for that train and testing
    dataset = CustomImageDataset(dataConfig)
    n = len(dataset)
    # 80:20 split
    split_index = int(n * 0.8)
    # Use random_split to create proper Subset objects that DataLoader accepts
    train_dataset, test_dataset = random_split(dataset, [split_index, n - split_index])

    # Pass the resulting Subset objects to the DataLoader
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)

    #create cnn model
    model = CNNModel()
    loss_fn = nn.CrossEntropyLoss()
    learning_rate = 0.001 
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Lists to store history for plotting
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []


    epochs = 5
    for t in range(epochs):
        print(f'Epoch {t+1}\n--------------------')
        avg_train_loss, train_acc = train(train_dataloader, model, loss_fn, optimizer)
        avg_test_loss, test_acc = test(test_dataloader, model, loss_fn)

        # Record metrics
        train_losses.append(avg_train_loss)
        train_accuracies.append(train_acc)
        test_losses.append(avg_test_loss)
        test_accuracies.append(test_acc)

    
    modelname = 'CNN_yash.pt'
    torch.save(model.state_dict(), modelname)
    print(f'Model trained and saved to {modelname} !')

    # Plot the results
    plt.figure(figsize=(10, 5))

    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs + 1), train_losses, label='Training Loss')
    plt.plot(range(1, epochs + 1), test_losses, label='Testing Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss over Epochs')
    plt.legend()
    plt.grid(True)

    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs + 1), train_accuracies, label='Training Accuracy')
    plt.plot(range(1, epochs + 1), test_accuracies, label='Testing Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy over Epochs')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show() # Display the graph window
    plt.savefig('training_progress.png') # Optional: save the graph to a file







