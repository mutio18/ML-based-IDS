#!/usr/bin/env python
"""
Test XGBoost Alone vs Voting Ensemble
Compare performance and see if ensemble adds value
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
from sklearn.metrics import recall_score, precision_score, f1_score, confusion_matrix

# ============================================
# LOAD MODELS
# ============================================
print("="*60)
print(" LOADING MODELS")
print("="*60)

# Load XGBoost model
xgb_model = joblib.load('models/UNSW/xgboost_latest.pkl')
print(" XGBoost model loaded")

# Load PyTorch model for comparison
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

device = torch.device('cpu')
pytorch_model = TabularMLP_19(input_dim=19)
pytorch_model.load_state_dict(torch.load('models/UNSW/pytorch_mlp_latest.pth', map_location=device))
pytorch_model.to(device)
pytorch_model.eval()
print(" PyTorch model loaded")

# ============================================
# LOAD TEST DATA
# ============================================
print("\n Loading UNSW test data...")
test_raw = pd.read_parquet('data/UNSW/UNSW_NB15_testing-set.parquet')
y_test = test_raw['label'].values
X_test = pd.read_csv('data/processed/X_test_optimized.csv')
print(f" Test data: {len(y_test)} samples, {X_test.shape[1]} features")

# ============================================
# GET PREDICTIONS
# ============================================
print("\n Getting predictions...")

# XGBoost only
xgb_probs = xgb_model.predict_proba(X_test.values)[:, 1]

# PyTorch only
pytorch_probs = []
batch_size = 5000
for i in range(0, len(X_test), batch_size):
    X_batch = X_test.iloc[i:i+batch_size]
    X_tensor = torch.FloatTensor(X_batch.values).to(device)
    with torch.no_grad():
        outputs = pytorch_model(X_tensor)
        probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
        pytorch_probs.extend(probs)
pytorch_probs = np.array(pytorch_probs)

# Ensemble
ensemble_probs = (xgb_probs + pytorch_probs) / 2

print(f" XGBoost predictions: {len(xgb_probs)}")
print(f" PyTorch predictions: {len(pytorch_probs)}")

# ============================================
# COMPARE AT DIFFERENT THRESHOLDS
# ============================================
print("\n" + "="*60)
print(" MODEL COMPARISON AT THRESHOLD 0.65")
print("="*60)

threshold = 0.65

# XGBoost
xgb_preds = (xgb_probs >= threshold).astype(int)
xgb_recall = recall_score(y_test, xgb_preds)
xgb_precision = precision_score(y_test, xgb_preds)
xgb_f1 = f1_score(y_test, xgb_preds)
xgb_fp = sum((xgb_preds == 1) & (y_test == 0))

# PyTorch
pytorch_preds = (pytorch_probs >= threshold).astype(int)
pytorch_recall = recall_score(y_test, pytorch_preds)
pytorch_precision = precision_score(y_test, pytorch_preds)
pytorch_f1 = f1_score(y_test, pytorch_preds)
pytorch_fp = sum((pytorch_preds == 1) & (y_test == 0))

# Ensemble
ensemble_preds = (ensemble_probs >= threshold).astype(int)
ensemble_recall = recall_score(y_test, ensemble_preds)
ensemble_precision = precision_score(y_test, ensemble_preds)
ensemble_f1 = f1_score(y_test, ensemble_preds)
ensemble_fp = sum((ensemble_preds == 1) & (y_test == 0))

print(f"\n{'Model':<15} {'Recall':<12} {'Precision':<12} {'F1':<12} {'False Alarms':<12}")
print("-" * 70)
print(f"{'XGBoost Only':<15} {xgb_recall:<12.4f} {xgb_precision:<12.4f} {xgb_f1:<12.4f} {xgb_fp:<12d}")
print(f"{'PyTorch Only':<15} {pytorch_recall:<12.4f} {pytorch_precision:<12.4f} {pytorch_f1:<12.4f} {pytorch_fp:<12d}")
print(f"{'Voting Ensemble':<15} {ensemble_recall:<12.4f} {ensemble_precision:<12.4f} {ensemble_f1:<12.4f} {ensemble_fp:<12d}")

# ============================================
# DETAILED COMPARISON - THRESHOLD RANGE
# ============================================
print("\n" + "="*60)
print(" COMPARISON ACROSS THRESHOLDS")
print("="*60)

thresholds = [0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]
results = []

for thresh in thresholds:
    xgb_pred = (xgb_probs >= thresh).astype(int)
    ensemble_pred = (ensemble_probs >= thresh).astype(int)
    
    results.append({
        'threshold': thresh,
        'xgb_recall': recall_score(y_test, xgb_pred),
        'xgb_precision': precision_score(y_test, xgb_pred),
        'ensemble_recall': recall_score(y_test, ensemble_pred),
        'ensemble_precision': precision_score(y_test, ensemble_pred),
        'xgb_fp': sum((xgb_pred == 1) & (y_test == 0)),
        'ensemble_fp': sum((ensemble_pred == 1) & (y_test == 0))
    })

print(f"\n{'Threshold':<10} {'XGB Recall':<12} {'XGB Prec':<12} {'XGB FP':<10} {'Ens Recall':<12} {'Ens Prec':<12} {'Ens FP':<10}")
print("-" * 85)

for r in results:
    print(f"{r['threshold']:<10.2f} {r['xgb_recall']:<12.4f} {r['xgb_precision']:<12.4f} {r['xgb_fp']:<10d} "
          f"{r['ensemble_recall']:<12.4f} {r['ensemble_precision']:<12.4f} {r['ensemble_fp']:<10d}")

# ============================================
# LIVE INTERACTIVE TEST
# ============================================
print("\n" + "="*60)
print("  LIVE TEST - XGBoost Only vs Ensemble")
print("="*60)
print("Enter connection details to compare predictions\n")

while True:
    try:
        print("-" * 40)
        proto = input("Protocol (tcp/udp/icmp) [tcp]: ").strip() or "tcp"
        service = input("Service (http/dns/ftp) [http]: ").strip() or "http"
        sbytes = int(input("Source bytes [500]: ").strip() or "500")
        dbytes = int(input("Destination bytes [1000]: ").strip() or "1000")
        rate = float(input("Rate [10]: ").strip() or "10")
        dur = float(input("Duration (seconds) [0]: ").strip() or "0")
        
        # Create features
        dangerous_protos = ['3pc', 'a/n', 'aes-sp3-d', 'any', 'argus']
        features = {
            'is_sm_ips_ports': 0,
            'sbytes': sbytes,
            'dbytes': dbytes,
            'rate': rate,
            'dur': dur,
            'sload': 0,
            'dload': 0,
            'sinpkt': 0,
            'dinpkt': 0,
            'sjit': 0,
            'djit': 0,
            'tcprtt': 0,
            'synack': 0,
            'ackdat': 0,
            'bytes_ratio': sbytes / (dbytes + 1),
            'packets_ratio': 1,
            'load_ratio': 1,
            'jitter_product': 0,
            'dangerous_proto': 1 if proto in dangerous_protos else 0
        }
        X = pd.DataFrame([features])
        
        # XGBoost prediction
        xgb_prob = xgb_model.predict_proba(X.values)[0][1]
        xgb_pred = "ATTACK" if xgb_prob >= 0.65 else "NORMAL"
        
        # PyTorch prediction
        X_tensor = torch.FloatTensor(X.values).to(device)
        with torch.no_grad():
            pytorch_output = pytorch_model(X_tensor)
            pytorch_prob = torch.softmax(pytorch_output, dim=1)[:, 1].cpu().numpy()[0]
        
        # Ensemble
        ensemble_prob = (xgb_prob + pytorch_prob) / 2
        ensemble_pred = "ATTACK" if ensemble_prob >= 0.65 else "NORMAL"
        
        print(f"\n RESULTS (threshold=0.65):")
        print(f"   XGBoost:    {xgb_prob:.2%} → {xgb_pred}")
        print(f"   PyTorch:    {pytorch_prob:.2%} → {'ATTACK' if pytorch_prob >= 0.65 else 'NORMAL'}")
        print(f"   Ensemble:   {ensemble_prob:.2%} → {ensemble_pred}")
        print(f"   XGBoost Confidence: {xgb_prob:.2%} | PyTorch: {pytorch_prob:.2%}")
        
    except KeyboardInterrupt:
        print("\nExiting...")
        break
    except Exception as e:
        print(f"Error: {e}")
        continue
    
    again = input("\nTest another? (y/n): ").strip().lower()
    if again != 'y':
        break

print("\n Done!")