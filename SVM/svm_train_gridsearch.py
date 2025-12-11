from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns


class_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L']

def train_svm_with_grid_search(X_train, y_train, X_test, y_test):
    from sklearn.model_selection import GridSearchCV
    
    # parameter grid
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 1, 0.1, 0.01, 0.001, 0.0001],
        'kernel': ['rbf', 'linear', 'poly']
    }
    
    svm = SVC(probability=True, random_state=42)
    
    # Grid search with cross-validation
    grid_search = GridSearchCV(
        svm, 
        param_grid, 
        cv=3, 
        n_jobs=-1, 
        verbose=2,
        scoring='accuracy'
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"Best grid search parameters: {grid_search.best_params_}")
    print(f"Best grid search CV score: {grid_search.best_score_:.4f}")
    
    # Get best model
    best_svm = grid_search.best_estimator_
    print(f"Best SVM: {best_svm}")
    
    # Evaluate on test set
    train_acc = best_svm.score(X_train, y_train)
    test_acc = best_svm.score(X_test, y_test)
    
    print(f"Train Accuracy: {train_acc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")

    grid_predictions = grid_search.predict(X_test) 
    print("Model evaluation")
    print(classification_report(y_test, grid_predictions, target_names=class_names))
    
    return best_svm, grid_search.best_params_, grid_search

def plot_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('SVM Grid Search Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('plots/svm_gridsearch_confusion_matrix.png')
    plt.close()
    print("Confusion matrix saved to 'plots/svm_gridsearch_confusion_matrix.png'")

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
    ax.set_title('SVM Performance with different Hyperparameters(Grid Search)')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Config {i+1}' for i in range(len(configs))], rotation=45)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plots/svm_gridsearch_hyperparameter_comparison.png')
    plt.close()
    print("Hyperparameter comparison saved to 'plots/svm_gridsearch_hyperparameter_comparison.png'")

if __name__ == '__main__':
    
    print("Loading features...")
    X = np.load('features/svm_features_X.npy')
    y = np.load('features/svm_features_y.npy')   
   
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("Training SVM Grid Search")
    
    best_svm, best_config, grid_search = train_svm_with_grid_search(
        X_train_scaled, y_train, X_test_scaled, y_test
    )
    
    # Extract top configurations from grid search for comparison plot
    cv_results = grid_search.cv_results_
    top_indices = np.argsort(cv_results['mean_test_score'])[-10:][::-1]  # Top 10 configs
    
    results = []
    for idx in top_indices:
        config = {
            'C': cv_results['param_C'][idx],
            'gamma': cv_results['param_gamma'][idx],
            'kernel': cv_results['param_kernel'][idx]
        }
        # Get train and test accuracy for this config
        model = grid_search.cv_results_['params'][idx]
        temp_svm = SVC(**model, probability=True, random_state=42)
        temp_svm.fit(X_train_scaled, y_train)
        
        results.append({
            'config': config,
            'train_acc': temp_svm.score(X_train_scaled, y_train),
            'test_acc': temp_svm.score(X_test_scaled, y_test)
        })
    
    print("Final Evaluation")
    
    y_train_pred = best_svm.predict(X_train_scaled)
    y_test_pred = best_svm.predict(X_test_scaled)
    
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    
    print(f"Final Train Accuracy: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
    print(f"Final Test Accuracy:  {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    
    with open('models/svm_gridsearch_model.pkl', 'wb') as f:
        pickle.dump(best_svm, f)
    with open('models/scaler_gridsearch.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    print("Model saved to models/svm_gridsearch_model.pkl")
    print("Scaler saved to models/scaler_gridsearch.pkl")
    
    plot_confusion_matrix(y_test, y_test_pred, class_names)
    if results:
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
    with open('results/svm_gridsearch_results.json', 'w') as f:
        json.dump(summary, f, indent=4)
    print("Results saved to 'results/svm_gridsearch_results.json'")
    
    print("SVM grid search train complete")
