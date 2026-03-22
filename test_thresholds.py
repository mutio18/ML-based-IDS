# test_thresholds_fixed.py
#!/usr/bin/env python
"""
Test Voting Ensemble with Different Thresholds
Using CORRECT UNSW test data
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
from sklearn.metrics import recall_score, precision_score, f1_score, confusion_matrix

# ============================================
# MODEL DEFINITION
# ============================================
class TabularMLP_19(nn.Module):
    def __init__(self, input_dim=19):
        super(TabularMLP_19, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.dropout3 = nn.Dropout(0.2)
        self.fc4 = nn.Linear(64, 2)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        x = self.relu(self.bn3(self.fc3(x)))
        x = self.dropout3(x)
        return self.fc4(x)

# ============================================
# LOAD MODELS
# ============================================
device = torch.device('cpu')
print("🔧 Loading Voting Ensemble...")

# Load PyTorch model
pytorch_model = TabularMLP_19(input_dim=19)
pytorch_model.load_state_dict(torch.load('models/UNSW/pytorch_mlp_latest.pth', map_location=device))
pytorch_model.to(device)
pytorch_model.eval()
print("✅ PyTorch model loaded")

# Load XGBoost model
xgb_model = joblib.load('models/UNSW/xgboost_latest.pkl')
print("✅ XGBoost model loaded")

# ============================================
# LOAD CORRECT UNSW TEST DATA
# ============================================
print("\n📊 Loading UNSW test data...")

# Load raw test data for labels
test_raw = pd.read_parquet('data/UNSW/UNSW_NB15_testing-set.parquet')
y_test = test_raw['label'].values

# Load preprocessed test features
X_test = pd.read_csv('data/processed/X_test_optimized.csv')

print(f"✅ Test data loaded: {len(y_test)} samples, {X_test.shape[1]} features")
print(f"   Features: {list(X_test.columns[:5])}...")

# Verify sizes match
if len(X_test) != len(y_test):
    print(f"⚠️ Mismatch! X_test: {len(X_test)}, y_test: {len(y_test)}")
    print("   Taking minimum size...")
    min_len = min(len(X_test), len(y_test))
    X_test = X_test.iloc[:min_len]
    y_test = y_test[:min_len]
    print(f"   Adjusted to: {len(y_test)} samples")

# ============================================
# GET ENSEMBLE PREDICTIONS
# ============================================
print("\n🔍 Getting ensemble predictions...")
all_probs = []
batch_size = 5000

for i in range(0, len(X_test), batch_size):
    X_batch = X_test.iloc[i:i+batch_size]
    X_tensor = torch.FloatTensor(X_batch.values).to(device)
    
    with torch.no_grad():
        pytorch_outputs = pytorch_model(X_tensor)
        pytorch_probs = torch.softmax(pytorch_outputs, dim=1)[:, 1].cpu().numpy()
    
    xgb_probs = xgb_model.predict_proba(X_batch.values)[:, 1]
    ensemble_probs = (pytorch_probs + xgb_probs) / 2
    all_probs.extend(ensemble_probs)
    
    print(f"   Processed {min(i+batch_size, len(X_test))}/{len(X_test)}")

all_probs = np.array(all_probs)
print(f"✅ Predictions ready for {len(all_probs)} samples")

# ============================================
# TEST DIFFERENT THRESHOLDS
# ============================================
print("\n" + "="*60)
print("📊 THRESHOLD COMPARISON")
print("="*60)

thresholds = [0.3, 0.4, 0.49, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
results = []

print(f"{'Threshold':<10} {'Recall':<10} {'Precision':<12} {'F1':<10} {'Attacks Flagged':<15} {'False Alarms':<12}")
print("-" * 80)

for thresh in thresholds:
    preds = (all_probs >= thresh).astype(int)
    recall = recall_score(y_test, preds)
    precision = precision_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    attacks_flagged = sum(preds)
    false_alarms = sum((preds == 1) & (y_test == 0))
    
    results.append({
        'threshold': thresh,
        'recall': recall,
        'precision': precision,
        'f1': f1,
        'attacks_flagged': attacks_flagged,
        'false_alarms': false_alarms
    })
    
    status = "✅" if recall >= 0.90 else "⚠️"
    print(f"{thresh:<10.2f} {recall:<10.4f} {precision:<12.4f} {f1:<10.4f} {attacks_flagged:<15d} {false_alarms:<12d} {status}")

# ============================================
# FIND BEST BALANCE
# ============================================
print("\n" + "="*60)
print("🏆 RECOMMENDATIONS")
print("="*60)

# Find thresholds that maintain recall >= 90%
good_thresholds = [r for r in results if r['recall'] >= 0.90]
if good_thresholds:
    # Best precision at recall >= 90%
    best_precision = max(good_thresholds, key=lambda x: x['precision'])
    print(f"\n🎯 Best precision while keeping recall ≥ 90%:")
    print(f"   Threshold: {best_precision['threshold']:.2f}")
    print(f"   Precision: {best_precision['precision']:.4f}")
    print(f"   Recall:    {best_precision['recall']:.4f}")
    print(f"   False Alarms: {best_precision['false_alarms']:,}")
    
    # Best F1
    best_f1 = max(good_thresholds, key=lambda x: x['f1'])
    print(f"\n🎯 Best F1-score while keeping recall ≥ 90%:")
    print(f"   Threshold: {best_f1['threshold']:.2f}")
    print(f"   F1: {best_f1['f1']:.4f}")
    print(f"   Precision: {best_f1['precision']:.4f}")
    print(f"   Recall: {best_f1['recall']:.4f}")
else:
    print("\n⚠️ No threshold achieves 90% recall")

# ============================================
# SUMMARY TABLE
# ============================================
print("\n" + "="*60)
print("📋 SUMMARY TABLE")
print("="*60)

summary_df = pd.DataFrame(results)
print(summary_df.to_string(index=False))

print("\n✅ Done!")