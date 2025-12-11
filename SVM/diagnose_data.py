import numpy as np
from sklearn.model_selection import train_test_split

# Load the features
X = np.load('features/svm_features_X.npy')
y = np.load('features/svm_features_y.npy')

print("=" * 60)
print("DIAGNOSTIC: Checking for data issues")
print("=" * 60)

# Basic stats
print(f"\nDataset shape: {X.shape}")
print(f"Labels shape: {y.shape}")
print(f"Number of features: {X.shape[1]}")
print(f"Number of samples: {X.shape[0]}")

# Check for duplicates
print("\n" + "=" * 60)
print("Checking for duplicate samples...")
print("=" * 60)
unique_samples = np.unique(X, axis=0)
print(f"Unique samples: {len(unique_samples)}")
print(f"Total samples: {len(X)}")
if len(unique_samples) < len(X):
    print(f"⚠️  WARNING: {len(X) - len(unique_samples)} duplicate samples found!")
else:
    print("✓ No duplicate samples")

# Check class distribution
print("\n" + "=" * 60)
print("Class distribution:")
print("=" * 60)
unique, counts = np.unique(y, return_counts=True)
for label, count in zip(unique, counts):
    print(f"Class {label}: {count} samples")

# Check if features are normalized/standardized
print("\n" + "=" * 60)
print("Feature statistics:")
print("=" * 60)
print(f"Feature mean: {X.mean():.4f}")
print(f"Feature std: {X.std():.4f}")
print(f"Feature min: {X.min():.4f}")
print(f"Feature max: {X.max():.4f}")

# Check for train/test leakage by splitting and checking overlap
print("\n" + "=" * 60)
print("Checking train/test split for leakage...")
print("=" * 60)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Convert to tuples for set comparison
train_tuples = set(map(tuple, X_train))
test_tuples = set(map(tuple, X_test))
overlap = train_tuples.intersection(test_tuples)

print(f"Train samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")
print(f"Overlapping samples: {len(overlap)}")

if len(overlap) > 0:
    print(f"⚠️  WARNING: {len(overlap)} samples appear in BOTH train and test sets!")
else:
    print("✓ No overlap between train and test sets")

# Check feature variance
print("\n" + "=" * 60)
print("Feature variance analysis:")
print("=" * 60)
feature_vars = X.var(axis=0)
zero_var_features = np.sum(feature_vars == 0)
low_var_features = np.sum(feature_vars < 0.001)
print(f"Features with zero variance: {zero_var_features}")
print(f"Features with very low variance (< 0.001): {low_var_features}")

# Sample some features to see what they look like
print("\n" + "=" * 60)
print("Sample features (first 5 samples, first 12 features):")
print("=" * 60)
print(X[:5, :12])

# Check if features are probability distributions (should sum to ~1)
print("\n" + "=" * 60)
print("Checking if features are probabilities (each 12-feature group should sum to ~1):")
print("=" * 60)
# Split into 6 groups of 12 features each
for i in range(6):
    start_idx = i * 12
    end_idx = start_idx + 12
    sums = X[:, start_idx:end_idx].sum(axis=1)
    print(f"Model {i+1} (features {start_idx}-{end_idx}): mean sum = {sums.mean():.4f}, std = {sums.std():.6f}")

print("\n" + "=" * 60)
print("ANALYSIS COMPLETE")
print("=" * 60)
