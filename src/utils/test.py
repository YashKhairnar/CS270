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
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()

    pred_list = []
    test_loss, correct = 0,0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            predicted_classes = pred.argmax(1)
        
            pred_list.append(predicted_classes)
            maxVote = vote(pred_list)

            if maxVote == y:
                correct += 1
            test_loss += loss_fn(pred, y).item()

    test_loss /= num_batches
    correct /= size

    accuracy_percent = correct * 100
    return test_loss, accuracy_percent


def vote(pred_list):
    counts = {}
    for pred in pred_list:
        counts[pred] = counts.get(pred, 0) + 1

    v = max(counts.values())
    for k, v in counts.items():
        if v == v: 
            return k