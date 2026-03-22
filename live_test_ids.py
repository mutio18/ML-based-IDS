#!/usr/bin/env python
"""
Live Testing for Voting Ensemble IDS
Test your model on custom network traffic data
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
import random
from datetime import datetime

# ============================================
# CONFIGURATION - ADD THIS SECTION RIGHT HERE
# ============================================
THRESHOLD = 0.65  # Options: 0.49, 0.60, 0.65, 0.70
# - 0.49: 98.7% recall, 75.9% precision (14,213 false alarms)
# - 0.65: 94.1% recall, 86.6% precision (6,623 false alarms)
# - 0.70: 90.3% recall, 90.1% precision (4,490 false alarms)
print(f" Using threshold: {THRESHOLD}")

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
print(" Loading Voting Ensemble...")

# Load PyTorch model
pytorch_model = TabularMLP_19(input_dim=19)
pytorch_model.load_state_dict(torch.load('models/UNSW/pytorch_mlp_latest.pth', map_location=device))
pytorch_model.to(device)
pytorch_model.eval()
print(" PyTorch model loaded")

# Load XGBoost model
xgb_model = joblib.load('models/UNSW/xgboost_latest.pkl')
print(" XGBoost model loaded")

# ============================================
# HELPER FUNCTIONS
# ============================================
def create_features_from_input(proto, service, sbytes, dbytes, rate, dur=0):
    """Create 19 features from basic inputs"""
    
    # Base features with defaults
    features = {
        'is_sm_ips_ports': 0,
        'sbytes': float(sbytes),
        'dbytes': float(dbytes),
        'rate': float(rate),
        'dur': float(dur),
        'sload': 0,
        'dload': 0,
        'sinpkt': 0,
        'dinpkt': 0,
        'sjit': 0,
        'djit': 0,
        'tcprtt': 0,
        'synack': 0,
        'ackdat': 0
    }
    
    # Engineered features
    features['bytes_ratio'] = features['sbytes'] / (features['dbytes'] + 1)
    features['packets_ratio'] = 1.0
    features['load_ratio'] = 1.0
    features['jitter_product'] = 0
    
    # Dangerous protocol indicator
    dangerous_protos = ['3pc', 'a/n', 'aes-sp3-d', 'any', 'argus']
    features['dangerous_proto'] = 1 if proto in dangerous_protos else 0
    
    return pd.DataFrame([features])

def predict_connection(proto, service, sbytes, dbytes, rate, dur=0):
    """Predict if a single connection is an attack"""
    
    # Create features
    X = create_features_from_input(proto, service, sbytes, dbytes, rate, dur)
    
    # PyTorch prediction
    X_tensor = torch.FloatTensor(X.values).to(device)
    with torch.no_grad():
        pytorch_output = pytorch_model(X_tensor)
        pytorch_prob = torch.softmax(pytorch_output, dim=1)[:, 1].cpu().numpy()[0]
    
    # XGBoost prediction
    xgb_prob = xgb_model.predict_proba(X.values)[0][1]
    
    # Ensemble
    ensemble_prob = (pytorch_prob + xgb_prob) / 2
    prediction = "ATTACK" if ensemble_prob >= THRESHOLD else "NORMAL"
    
    return {
        'prediction': prediction,
        'confidence': ensemble_prob,
        'pytorch_confidence': pytorch_prob,
        'xgb_confidence': xgb_prob,
        'attack_probability': ensemble_prob
    }

def generate_random_traffic():
    """Generate random test traffic for simulation"""
    
    protocols = ['tcp', 'udp', 'icmp', 'http', 'https', 'ftp', 'smtp', 'dns']
    services = ['http', 'dns', 'ftp', 'smtp', 'telnet', 'ssh', 'private']
    flags = ['SF', 'S0', 'REJ', 'RSTO']
    
    proto = random.choice(protocols)
    service = random.choice(services)
    sbytes = random.randint(0, 10000)
    dbytes = random.randint(0, 20000)
    rate = random.uniform(0, 100)
    dur = random.uniform(0, 10)
    
    return proto, service, sbytes, dbytes, rate, dur

# ============================================
# TEST MENU
# ============================================
def interactive_test():
    """Interactive testing mode"""
    print("\n" + "="*60)
    print("  LIVE IDS TESTING - INTERACTIVE MODE")
    print("="*60)
    print("Enter connection details or type 'random' for random test")
    print("Type 'batch' for batch test, 'quit' to exit\n")
    
    while True:
        try:
            print("-" * 40)
            choice = input("Mode (manual/random/batch/quit): ").strip().lower()
            
            if choice == 'quit':
                break
                
            elif choice == 'random':
                proto, service, sbytes, dbytes, rate, dur = generate_random_traffic()
                print(f"\n📡 Random Connection Generated:")
                print(f"   Protocol: {proto}, Service: {service}")
                print(f"   sbytes: {sbytes}, dbytes: {dbytes}")
                print(f"   rate: {rate:.2f}, dur: {dur:.2f}")
                
                result = predict_connection(proto, service, sbytes, dbytes, rate, dur)
                print(f"\n RESULT: {result['prediction']}")
                print(f"   Attack Probability: {result['confidence']:.2%}")
                print(f"   PyTorch: {result['pytorch_confidence']:.2%} | XGBoost: {result['xgb_confidence']:.2%}")
                
            elif choice == 'batch':
                print("\n Running 10 random tests...")
                print("-" * 60)
                attacks = 0
                for i in range(10):
                    proto, service, sbytes, dbytes, rate, dur = generate_random_traffic()
                    result = predict_connection(proto, service, sbytes, dbytes, rate, dur)
                    attacks += 1 if result['prediction'] == 'ATTACK' else 0
                    status = " ATTACK" if result['prediction'] == 'ATTACK' else " NORMAL"
                    print(f"  {i+1:2d}. {proto:6s} {service:8s} bytes={sbytes:5d}→{dbytes:5d} | {status} ({result['confidence']:.1%})")
                print(f"\n Summary: {attacks}/10 connections classified as ATTACK")
                
            elif choice == 'manual':
                print("\n Enter Connection Details:")
                proto = input("   Protocol (tcp/udp/icmp) [tcp]: ").strip() or "tcp"
                service = input("   Service (http/dns/ftp) [http]: ").strip() or "http"
                sbytes = int(input("   Source bytes [500]: ").strip() or "500")
                dbytes = int(input("   Destination bytes [1000]: ").strip() or "1000")
                rate = float(input("   Rate [10]: ").strip() or "10")
                dur = float(input("   Duration (seconds) [0]: ").strip() or "0")
                
                result = predict_connection(proto, service, sbytes, dbytes, rate, dur)
                print(f"\n RESULT: {result['prediction']}")
                print(f"   Attack Probability: {result['confidence']:.2%}")
                print(f"   PyTorch Confidence: {result['pytorch_confidence']:.2%}")
                print(f"   XGBoost Confidence: {result['xgb_confidence']:.2%}")
                
            else:
                print(" Unknown mode. Try: manual, random, batch, quit")
                
        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except Exception as e:
            print(f" Error: {e}")

def file_test(filepath):
    """Test on a CSV file"""
    print(f"\n Testing file: {filepath}")
    df = pd.read_csv(filepath)
    print(f"Found {len(df)} connections\n")
    
    results = []
    for idx, row in df.iterrows():
        # Use columns if available, otherwise use defaults
        proto = row.get('proto', 'tcp')
        service = row.get('service', 'http')
        sbytes = row.get('sbytes', 500)
        dbytes = row.get('dbytes', 1000)
        rate = row.get('rate', 10)
        dur = row.get('dur', 0)
        
        result = predict_connection(proto, service, sbytes, dbytes, rate, dur)
        results.append(result)
        
        # Print progress
        if (idx + 1) % 10 == 0:
            print(f"   Processed {idx+1}/{len(df)}...")
    
    # Summary
    attacks = sum(1 for r in results if r['prediction'] == 'ATTACK')
    print(f"\n FILE TEST SUMMARY:")
    print(f"   Total connections: {len(results)}")
    print(f"   Attacks detected: {attacks} ({attacks/len(results)*100:.1f}%)")
    print(f"   Normal traffic:   {len(results)-attacks} ({(len(results)-attacks)/len(results)*100:.1f}%)")
    
    return results

# ============================================
# MAIN
# ============================================
if __name__ == "__main__":
    import sys
    
    print("="*60)
    print(" VOTING ENSEMBLE IDS - LIVE TESTING")
    print("="*60)
    print(f"Model: PyTorch + XGBoost")
    print(f"Expected Recall: 98.7%")
    print(f"Expected Precision: 75.9%")
    
    if len(sys.argv) > 1:
        # File mode
        file_test(sys.argv[1])
    else:
        # Interactive mode
        interactive_test()