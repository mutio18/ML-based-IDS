#!/usr/bin/env python
"""
PRODUCTION INTRUSION DETECTION SYSTEM
Voting Ensemble: PyTorch MLP + XGBoost
Optimized Threshold: 0.681

Performance on UNSW-NB15 Test Set:
- Precision: 88.8%
- Recall: 92.1% (above 90% target)
- F1-Score: 90.4%
- False Alarms Reduced: 47%
"""

import joblib
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import json
import os
import sys
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ============================================
# CONFIGURATION
# ============================================
MODEL_DIR = Path('models/UNSW')
THRESHOLD = 0.681  # Optimized threshold from analysis
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============================================
# 1. MODEL ARCHITECTURE
# ============================================
class TabularMLP(nn.Module):
    """MLP model - 19 input features"""
    def __init__(self, input_dim=19):
        super(TabularMLP, self).__init__()
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
# 2. PRODUCTION DETECTOR CLASS
# ============================================
class ProductionIDS:
    """Production-ready Intrusion Detection System"""
    
    def __init__(self, threshold=THRESHOLD):
        self.threshold = threshold
        self.device = DEVICE
        
        print("="*60)
        print(" PRODUCTION IDS INITIALIZING")
        print("="*60)
        print(f"Threshold: {self.threshold}")
        print(f"Device: {self.device}")
        
        self._load_models()
        self._init_features()
        self._print_performance()
    
    def _load_models(self):
        """Load trained models"""
        try:
            # Load PyTorch model
            self.pytorch_model = TabularMLP()
            pytorch_path = MODEL_DIR / 'pytorch_mlp_latest.pth'
            self.pytorch_model.load_state_dict(
                torch.load(pytorch_path, map_location=self.device)
            )
            self.pytorch_model.to(self.device)
            self.pytorch_model.eval()
            print(" PyTorch model loaded")
            
            # Load XGBoost model
            xgb_path = MODEL_DIR / 'xgboost_latest.pkl'
            self.xgb_model = joblib.load(xgb_path)
            print(" XGBoost model loaded")
            
        except Exception as e:
            print(f" Error loading models: {e}")
            sys.exit(1)
    
    def _init_features(self):
        """Initialize feature lists"""
        self.base_features = [
            'is_sm_ips_ports', 'sbytes', 'dbytes', 'rate', 'dur',
            'sload', 'dload', 'sinpkt', 'dinpkt', 'sjit', 'djit',
            'tcprtt', 'synack', 'ackdat'
        ]
        
        self.dangerous_protocols = ['3pc', 'a/n', 'aes-sp3-d', 'any', 'argus']
        self.all_features = self.base_features + [
            'bytes_ratio', 'packets_ratio', 'load_ratio', 
            'jitter_product', 'dangerous_proto'
        ]
    
    def _print_performance(self):
        """Display expected performance"""
        print("\n EXPECTED PERFORMANCE:")
        print(f"   Precision: 88.8%")
        print(f"   Recall:    92.1%")
        print(f"   F1-Score:  90.4%")
        print(f"   Threshold: {self.threshold}")
    
    def engineer_features(self, df):
        """Create engineered features from raw input"""
        X = df[self.base_features].copy()
        
        # Add engineered features
        X['bytes_ratio'] = df['sbytes'] / (df['dbytes'] + 1)
        X['packets_ratio'] = df.get('spkts', 0) / (df.get('dpkts', 0) + 1)
        X['load_ratio'] = df['sload'] / (df['dload'] + 1)
        X['jitter_product'] = df['sjit'] * df['djit']
        
        # Dangerous protocol indicator
        X['dangerous_proto'] = df.get('proto', '').isin(self.dangerous_protocols).astype(int)
        
        return X[self.all_features]
    
    def predict_proba(self, df):
        """Get ensemble probability of attack"""
        # Engineer features
        X = self.engineer_features(df)
        X_np = X.values.astype(np.float32)
        
        # PyTorch prediction
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_np).to(self.device)
            pytorch_outputs = self.pytorch_model(X_tensor)
            pytorch_probs = torch.softmax(pytorch_outputs, dim=1)[:, 1].cpu().numpy()
        
        # XGBoost prediction
        xgb_probs = self.xgb_model.predict_proba(X_np)[:, 1]
        
        # Ensemble average
        ensemble_probs = (pytorch_probs + xgb_probs) / 2
        
        return ensemble_probs
    
    def predict(self, df):
        """Make binary prediction using threshold"""
        probs = self.predict_proba(df)
        return (probs >= self.threshold).astype(int)
    
    def predict_single(self, connection_dict):
        """Predict single connection from dictionary"""
        df = pd.DataFrame([connection_dict])
        prob = self.predict_proba(df)[0]
        pred = 'ATTACK' if prob >= self.threshold else 'NORMAL'
        
        return {
            'timestamp': datetime.now().isoformat(),
            'prediction': pred,
            'attack_probability': float(prob),
            'confidence': float(prob),
            'threshold_used': self.threshold
        }
    
    def predict_file(self, input_path, output_path=None):
        """Predict on entire CSV file"""
        print(f"\n Processing: {input_path}")
        df = pd.read_csv(input_path)
        print(f"   Found {len(df)} connections")
        
        # Get predictions
        probs = self.predict_proba(df)
        preds = (probs >= self.threshold).astype(int)
        
        # Create results
        results = df.copy()
        results['attack_probability'] = probs
        results['prediction'] = ['ATTACK' if p == 1 else 'NORMAL' for p in preds]
        results['confidence'] = probs
        
        # Summary
        n_attacks = sum(preds)
        n_normal = len(preds) - n_attacks
        print(f"\n RESULTS SUMMARY:")
        print(f"   Attacks detected: {n_attacks} ({n_attacks/len(preds)*100:.1f}%)")
        print(f"   Normal traffic:   {n_normal} ({n_normal/len(preds)*100:.1f}%)")
        
        # Save
        if output_path:
            results.to_csv(output_path, index=False)
            print(f" Results saved to: {output_path}")
        
        return results
    
    def interactive_mode(self):
        """Interactive prediction mode"""
        print("\n" + "="*60)
        print("  INTERACTIVE DETECTION MODE")
        print("="*60)
        print(f"Threshold: {self.threshold}")
        print("(Enter 'quit' to exit)")
        
        while True:
            print("\n" + "-"*40)
            try:
                # Get user input
                sbytes = float(input("Source bytes [500]: ") or "500")
                dbytes = float(input("Destination bytes [1000]: ") or "1000")
                rate = float(input("Rate [10]: ") or "10")
                proto = input("Protocol [tcp]: ").strip() or "tcp"
                
                # Build feature dict with defaults
                features = {
                    'is_sm_ips_ports': 0,
                    'sbytes': sbytes,
                    'dbytes': dbytes,
                    'rate': rate,
                    'dur': 0,
                    'sload': 0,
                    'dload': 0,
                    'sinpkt': 0,
                    'dinpkt': 0,
                    'sjit': 0,
                    'djit': 0,
                    'tcprtt': 0,
                    'synack': 0,
                    'ackdat': 0,
                    'proto': proto,
                    'spkts': 1,
                    'dpkts': 1
                }
                
                result = self.predict_single(features)
                print(f"\n RESULT: {result['prediction']}")
                print(f"   Attack Probability: {result['attack_probability']:.4f}")
                print(f"   Confidence: {result['confidence']*100:.1f}%")
                
            except KeyboardInterrupt:
                print("\n\nExiting...")
                break
            except Exception as e:
                print(f" Error: {e}")
                continue
            
            again = input("\nTest another? (y/n): ").lower()
            if again != 'y':
                break
    
    def get_model_info(self):
        """Return model information"""
        return {
            'name': 'Voting Ensemble (PyTorch + XGBoost)',
            'threshold': self.threshold,
            'features': self.all_features,
            'performance': {
                'precision': 0.888,
                'recall': 0.921,
                'f1_score': 0.904
            },
            'dangerous_protocols': self.dangerous_protocols
        }


# ============================================
# 3. COMMAND LINE INTERFACE
# ============================================
def main():
    import argparse
    parser = argparse.ArgumentParser(description='Production IDS Detector')
    parser.add_argument('--threshold', type=float, default=THRESHOLD,
                       help=f'Detection threshold (default: {THRESHOLD})')
    parser.add_argument('--file', '-f', type=str,
                       help='CSV file to analyze')
    parser.add_argument('--output', '-o', type=str,
                       help='Output file for results')
    parser.add_argument('--interactive', '-i', action='store_true',
                       help='Run in interactive mode')
    parser.add_argument('--info', action='store_true',
                       help='Show model information')
    
    args = parser.parse_args()
    
    # Initialize detector
    detector = ProductionIDS(threshold=args.threshold)
    
    if args.info:
        info = detector.get_model_info()
        print("\n MODEL INFORMATION:")
        print(json.dumps(info, indent=2))
    
    elif args.interactive:
        detector.interactive_mode()
    
    elif args.file:
        detector.predict_file(args.file, args.output)
    
    else:
        print("\n Usage examples:")
        print("  python production_ids.py --interactive")
        print("  python production_ids.py --file traffic.csv --output results.csv")
        print("  python production_ids.py --info")
        print("  python production_ids.py --threshold 0.7 --interactive")


if __name__ == "__main__":
    main()