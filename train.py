# train.py (Revised)
import torch

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    
    current_loss = 0
    correct = 0 # To calculate accuracy during training

    for batch, (X, y) in enumerate(dataloader):
        optimizer.zero_grad() 

        pred = model(X)
        loss = loss_fn(pred, y)

        loss.backward()
        optimizer.step()
        
        # Accumulate loss
        current_loss += loss.item()

        # Calculate Accuracy during training
        predicted_classes = pred.argmax(1)
        correct += (predicted_classes == y).type(torch.float).sum().item()

        if batch % 100 == 0:
            loss_item = loss.item()
            current = (batch + 1) * len(X)
            print(f'loss: {loss_item:>7f} [{current:>5d}/{size:>5d}]')
    
    # Return average loss and accuracy for the epoch
    avg_loss = current_loss / len(dataloader)
    accuracy = (correct / size) * 100
    return avg_loss, accuracy
