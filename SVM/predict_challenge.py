import numpy as np
import pickle

# Label mapping: 0-11 -> A-L
LABEL_MAP = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D',
    4: 'E', 5: 'F', 6: 'G', 7: 'H',
    8: 'I', 9: 'J', 10: 'K', 11: 'L'
}

def main():
    print("="*60)
    print("Running SVM inference on challenge dataset")
    print("="*60)
    
    # Load challenge features
    print("\nLoading challenge features...")
    X = np.load('challenge_features.npy')
    sample_ids = np.load('challenge_sample_ids.npy')
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Number of samples: {len(sample_ids)}")
    
    # Load trained SVM model and scaler
    print("\nLoading trained SVM model...")
    with open('models/svm_gridsearch_model.pkl', 'rb') as f:
        svm_model = pickle.load(f)
    
    print("Loading scaler...")
    with open('models/scaler_gridsearch.pkl', 'rb') as f:
        scaler = pickle.load(f)
    
    # Scale features
    print("\nScaling features...")
    X_scaled = scaler.transform(X)
    
    # Run predictions
    print("Running SVM predictions...")
    predictions = svm_model.predict(X_scaled)
    
    print(f"Predictions shape: {predictions.shape}")
    print(f"Unique labels predicted: {sorted(set(predictions))}")
    
    # Convert to letter labels
    letter_predictions = [LABEL_MAP[pred] for pred in predictions]
    
    # Create submission file
    print("\nGenerating submission file...")
    with open('challenge_predictions.txt', 'w') as f:
        for sample_id, label in zip(sample_ids, letter_predictions):
            f.write(f"{sample_id}, {label}\n")
    
    print("Submission file saved to 'challenge_predictions.txt'")
    
    # Show sample predictions
    print("\n" + "="*60)
    print("Sample predictions (first 10):")
    print("="*60)
    for i in range(min(10, len(sample_ids))):
        print(f"{sample_ids[i]}, {letter_predictions[i]}")
    
    print("\n" + "="*60)
    print("Sample predictions (last 10):")
    print("="*60)
    for i in range(max(0, len(sample_ids)-10), len(sample_ids)):
        print(f"{sample_ids[i]}, {letter_predictions[i]}")
    
    # Show prediction distribution
    print("\n" + "="*60)
    print("Prediction distribution:")
    print("="*60)
    from collections import Counter
    pred_counts = Counter(letter_predictions)
    for label in sorted(pred_counts.keys()):
        count = pred_counts[label]
        percentage = (count / len(letter_predictions)) * 100
        print(f"Class {label}: {count:4d} samples ({percentage:5.2f}%)")
    
    print("\n" + "="*60)
    print("Classification complete!")
    print("="*60)
    print(f"Total samples classified: {len(sample_ids)}")
    print(f"Output file: challenge_predictions.txt")

if __name__ == '__main__':
    main()
