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
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

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

    # for early stopping
    best_acc = 0.0               # best validation/test accuracy seen so far
    patience = 3                 # how many epochs to wait with no improvement
    epochs_no_improve = 0
    best_model_path = "yash_best_model.pth"  # filename to save best model
    min_delta = 0.0              # minimum change in accuracy to count as improvement

    epochs = 5
    for t in range(epochs):
        print(f"Epoch {t+1}/{epochs}")
        avg_train_loss, train_acc = train(train_dataloader, model, loss_fn, optimizer)
        avg_test_loss, test_acc = test(test_dataloader, model, loss_fn)

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

    
    # ---------- Load best model before final save ----------
    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    modelname = 'CNN_yash.pt'
    torch.save(model.state_dict(), modelname)
    print(f"\nBest model weights saved to {modelname} (best acc = {checkpoint['best_acc']:.2f}%)")


     # ---------- Plot training curves ----------
    epochs_run = len(train_losses)

    plt.figure(figsize=(10, 5))

    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs_run + 1), train_losses, label='Training Loss')
    plt.plot(range(1, epochs_run + 1), test_losses, label='Testing Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss over Epochs')
    plt.legend()
    plt.grid(True)

    # Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs_run + 1), train_accuracies, label='Training Accuracy')
    plt.plot(range(1, epochs_run + 1), test_accuracies, label='Testing Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy over Epochs')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('training_progress.png')
    plt.show()