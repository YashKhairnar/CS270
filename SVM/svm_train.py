from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns


def train_svm_manual(X_train, y_train, X_test, y_test):
    # test different configs
    configs = [
        {'kernel': 'rbf', 'C': 1, 'gamma': 'scale'},
        {'kernel': 'rbf', 'C': 10, 'gamma': 'scale'},
        {'kernel': 'rbf', 'C': 100, 'gamma': 'scale'},
        {'kernel': 'rbf', 'C': 10, 'gamma': 0.001},
        {'kernel': 'rbf', 'C': 10, 'gamma': 0.01},
        {'kernel': 'linear', 'C': 1},
        {'kernel': 'linear', 'C': 10},
        {'kernel': 'poly', 'C': 1, 'degree': 3, 'gamma': 'scale'},
    ]
    
    best_accuracy = 0
    best_config = None
    best_model = None
    
    results = []
    
    print("Testing SVM configurations:")
    
    for i, config in enumerate(configs, 1):
        print(f"Config {i}/{len(configs)}: {config}")
        
        svm = SVC(**config, random_state=42, probability=True)
        svm.fit(X_train, y_train)
        
        train_acc = svm.score(X_train, y_train)
        test_acc = svm.score(X_test, y_test)
        
        print(f"Train accuracy: {train_acc:.4f}")
        print(f"Test accuracy:  {test_acc:.4f}")
        
        results.append({
            'config': config,
            'train_acc': train_acc,
            'test_acc': test_acc
        })
        
        if test_acc > best_accuracy:
            best_accuracy = test_acc
            best_config = config
            best_model = svm
            print(f"New best model!")
    
    print(f"Best config: {best_config}")
    print(f"Best test accuracy: {best_accuracy:.4f}")

    
    return best_model, best_config, results

def plot_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('SVM Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('svm_confusion_matrix.png')
    plt.close()
    print("Confusion matrix saved to 'svm_confusion_matrix.png'")

def plot_results_comparison(results):
    configs = [str(r['config']) for r in results]
    train_accs = [r['train_acc'] for r in results]
    test_accs = [r['test_acc'] for r in results]
    
    x = np.arange(len(configs))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x - width/2, train_accs, width, label='Train Accuracy')
    ax.bar(x + width/2, test_accs, width, label='Test Accuracy')
    
    ax.set_xlabel('Configuration')
    ax.set_ylabel('Accuracy')
    ax.set_title('SVM Performance with different Hyperparameters')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Config {i+1}' for i in range(len(configs))], rotation=45)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('svm_hyperparameter_comparison.png')
    plt.close()
    print("Hyperparameter comparison saved to 'svm_hyperparameter_comparison.png'")

if __name__ == '__main__':
    print("Load features")
    X = np.load('svm_features_X.npy')
    y = np.load('svm_features_y.npy')   
   
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("Training SVM...")
    
    best_svm, best_config, results = train_svm_manual(
        X_train_scaled, y_train, X_test_scaled, y_test
    )
    
    print("Final Evaluation")
    
    y_train_pred = best_svm.predict(X_train_scaled)
    y_test_pred = best_svm.predict(X_test_scaled)
    
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    
    print(f"Final Train Accuracy: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
    print(f"Final Test Accuracy:  {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    
    print("Classification Report:")
    class_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L']
    print(classification_report(y_test, y_test_pred, target_names=class_names))
    
    print("\nSaving model and scaler...")
    with open('svm_model.pkl', 'wb') as f:
        pickle.dump(best_svm, f)
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    print("Model saved to svm_model.pkl")
    print("Scaler saved to scaler.pkl")
    
    plot_confusion_matrix(y_test, y_test_pred, class_names)
    if 'results' in locals():
        plot_results_comparison(results)
    
    summary = {
        'best_hyperparameters': best_config,
        'train_accuracy': float(train_accuracy),
        'test_accuracy': float(test_accuracy),
        'feature_dim': X.shape[1],
        'num_train_samples': X_train.shape[0],
        'num_test_samples': X_test.shape[0]
    }
    
    import json
    with open('svm_results.json', 'w') as f:
        json.dump(summary, f, indent=4)
    print("Results saved to 'svm_results.json'")
    
    print("SVM train complete")
