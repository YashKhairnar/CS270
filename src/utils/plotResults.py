import matplotlib.pyplot as plt
import os

def plotResults(train_losses, test_losses, train_accuracies, test_accuracies, model_name, malware_type):
    epochs_run = len(train_losses)
    
    # Get final values
    final_train_loss = train_losses[-1]
    final_test_loss = test_losses[-1]
    final_train_acc = train_accuracies[-1]
    final_test_acc = test_accuracies[-1]
    
    plt.figure(figsize=(12, 5))

    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs_run + 1), train_losses, label='Training Loss')
    plt.plot(range(1, epochs_run + 1), test_losses, label='Testing Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss over Epochs')
    plt.legend()
    plt.grid(True)
    
    # Add final values as text annotation
    plt.text(0.02, 0.98, f'Final Train Loss: {final_train_loss:.4f}\nFinal Test Loss: {final_test_loss:.4f}',
             transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs_run + 1), train_accuracies, label='Training Accuracy')
    plt.plot(range(1, epochs_run + 1), test_accuracies, label='Testing Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy over Epochs')
    plt.legend()
    plt.grid(True)
    
    # Add final values as text annotation
    plt.text(0.02, 0.98, f'Final Train Acc: {final_train_acc:.2f}%\nFinal Test Acc: {final_test_acc:.2f}%',
             transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

    plt.tight_layout()
    
    os.makedirs('../plots', exist_ok=True)
    plot_path = os.path.join('../plots', f'{model_name}_{malware_type}_training_progress.png')
    plt.savefig(plot_path)
    print(f"Training plot saved to {plot_path}")
    plt.show()