import pandas as pd
import joblib
import numpy as np
from sklearn.model_selection import cross_val_score

print("🔍 DIAGNOSING YOUR MODEL")
print("="*60)

# Load data
X_train = pd.read_csv('data/processed/X_train_final.csv')
y_train = pd.read_csv('data/processed/y_train.csv').values.ravel()
model = joblib.load('models/best_model_latest.pkl')

print(f"Training data shape: {X_train.shape}")
print(f"Model type: {type(model).__name__}")

# Check class distribution in training
normal_count = sum(y_train == 0)
attack_count = sum(y_train == 1)
print(f"\n📊 Training Class Distribution:")
print(f"   Normal: {normal_count} ({normal_count/len(y_train)*100:.1f}%)")
print(f"   Attack: {attack_count} ({attack_count/len(y_train)*100:.1f}%)")

# Check KNN parameters
if hasattr(model, 'n_neighbors'):
    print(f"\n🔧 KNN Parameters:")
    print(f"   k (neighbors): {model.n_neighbors}")
    print(f"   weights: {model.weights}")
    print(f"   metric: {model.metric}")

# Quick cross-validation to see if model is consistent
scores = cross_val_score(model, X_train, y_train, cv=3, scoring='f1')
print(f"\n📈 Cross-validation F1 scores: {scores}")
print(f"   Mean F1: {scores.mean():.4f}")

print("\n✅ Diagnosis complete!")