import torch

def test(dataloader, model, loss_fn, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval() 

    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            
            predicted_classes = pred.argmax(1)
            correct += (predicted_classes == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size

    # Return average loss and accuracy for the epoch
    accuracy_percent = correct * 100
    return test_loss, accuracy_percent



def maxVoting(dataloader, model, loss_fn, device):
    """
    Evaluate model using voting across 6 image types per sample.
    Each sample in the dataloader contains 6 images (types a-f).
    """
    size = len(dataloader.dataset)
    model.eval()
    
    test_loss = 0.0
    correct = 0
    
    with torch.no_grad():
        for images_batch, labels_batch in dataloader:
            # images_batch shape: (batch_size, 6, C, H, W)
            # labels_batch shape: (batch_size,)
            
            batch_size = images_batch.size(0)
            
            for i in range(batch_size):
                # Get all 6 images for this sample
                six_images = images_batch[i]  # Shape: (6, C, H, W)
                true_label = labels_batch[i]
                
                # Move to device
                six_images = six_images.to(device)
                true_label = true_label.to(device)
                
                # Get predictions for all 6 images
                predictions = model(six_images)  # Shape: (6, num_classes)
                predicted_classes = predictions.argmax(1)  # Shape: (6,)
                
                # Vote across the 6 predictions
                voted_class = vote(predicted_classes)
                
                # Check if voted class matches true label
                if voted_class == true_label.item():
                    correct += 1
                
                # Calculate loss (average across 6 images)
                loss = loss_fn(predictions, true_label.expand(6))
                test_loss += loss.item()
    
    avg_loss = test_loss / size
    accuracy_percent = (correct / size) * 100
    
    return avg_loss, accuracy_percent


def vote(pred_list):
    """
    Find the class with the most votes. Handle ties randomly.
    
    Args:
        pred_list: Tensor of predictions (shape: 6,)
    
    Returns:
        The class with the most votes (int)
    """
    # Count votes for each class
    counts = {}
    for pred in pred_list.tolist():
        counts[pred] = counts.get(pred, 0) + 1
    
    # Find the maximum vote count
    max_votes = max(counts.values())
    
    # Get all classes with max votes (for tie-breaking)
    max_classes = [k for k, v in counts.items() if v == max_votes]
    
    # If there's a tie, randomly select one
    if len(max_classes) > 1:
        random_idx = torch.randperm(len(max_classes))[0].item()
        return max_classes[random_idx]
    else:
        return max_classes[0]