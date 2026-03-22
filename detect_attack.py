#!/usr/bin/env python
"""
Network Intrusion Detection System - Detection Tool
Loads trained model and detects attacks in network traffic data
"""

import pandas as pd
import numpy as np
import joblib
import argparse
import os
import sys
from datetime import datetime
from pathlib import Path


# CONFIGURATION
MODEL_PATH = 'models/best_model_latest.pkl'
ENCODER_PATH = 'models/encoders.pkl'
SCALER_PATH = 'models/scaler.pkl'
FEATURES_PATH = 'data/processed/selected_features.csv'

# Column names for raw data (if needed)
COL_NAMES = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
    'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
    'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
    'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
    'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
    'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
    'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
    'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'label', 'difficulty'
]

# List of all feature columns the model expects (excluding label/difficulty)
FEATURE_COLUMNS = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
    'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
    'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
    'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
    'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
    'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
    'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
    'dst_host_rerror_rate', 'dst_host_srv_rerror_rate'
]

# LOAD MODEL AND PREPROCESSORS
def load_artifacts():
    """Load the trained model, encoders, and scaler"""
    print("🔧 Loading detection model...")
    
    # Check if files exist
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model not found at {MODEL_PATH}")
        print("   Please train a model first or check the path.")
        return None, None, None, None
    
    try:
        model = joblib.load(MODEL_PATH)
        encoders = joblib.load(ENCODER_PATH) if os.path.exists(ENCODER_PATH) else None
        scaler = joblib.load(SCALER_PATH) if os.path.exists(SCALER_PATH) else None
        
        # Load selected features if available
        if os.path.exists(FEATURES_PATH):
            features_df = pd.read_csv(FEATURES_PATH)
            selected_features = features_df['selected_features'].tolist()
        else:
            selected_features = None
        
        print(f"✅ Model loaded: {type(model).__name__}")
        print(f"✅ Encoders loaded: {encoders is not None}")
        print(f"✅ Scaler loaded: {scaler is not None}")
        
        return model, encoders, scaler, selected_features
    
    except Exception as e:
        print(f"❌ Error loading artifacts: {e}")
        return None, None, None, None


# PREPROCESS SINGLE RECORD (FIXED VERSION)
def preprocess_record(record, encoders, scaler, selected_features):
    """Preprocess a single network connection record"""
    
    # Convert to DataFrame if it's a dictionary or list
    if isinstance(record, dict):
        df = pd.DataFrame([record])
    elif isinstance(record, list):
        # Assume list is in correct order
        df = pd.DataFrame([record], columns=COL_NAMES[:len(record)])
    elif isinstance(record, pd.DataFrame):
        df = record
    else:
        raise ValueError("Record must be dict, list, or DataFrame")
    
    # Make a copy
    df_processed = df.copy()
    
    # Ensure all required feature columns exist with default values
    for col in FEATURE_COLUMNS:
        if col not in df_processed.columns:
            # Add default values based on column type
            if col in ['protocol_type', 'service', 'flag']:
                df_processed[col] = 'unknown'
            else:
                df_processed[col] = 0
    
    # Handle categorical encoding
    if encoders:
        for col, le in encoders.items():
            if col in df_processed.columns:
                # Convert to string first
                df_processed[col] = df_processed[col].astype(str)
                # Map to encoded values, use -1 for unseen categories
                df_processed[col] = df_processed[col].map(
                    lambda x: le.transform([x])[0] if x in le.classes_ else -1
                )
    
    # Ensure correct column order for scaling
    df_processed = df_processed[FEATURE_COLUMNS]
    
    # Handle scaling
    if scaler:
        # Get numerical columns (exclude categorical ones)
        if encoders:
            cat_cols = list(encoders.keys())
            num_cols = [col for col in FEATURE_COLUMNS if col not in cat_cols]
        else:
            num_cols = FEATURE_COLUMNS
        
        if num_cols:
            df_processed[num_cols] = scaler.transform(df_processed[num_cols])
    
    # Select only the features used in training (if selected_features provided)
    if selected_features is not None:
        # Only keep columns that are in selected_features
        available_features = [f for f in selected_features if f in df_processed.columns]
        df_processed = df_processed[available_features]
    
    return df_processed

# DETECT ATTACKS
def detect_attack(record, model, encoders, scaler, selected_features):
    """Detect if a network connection is an attack"""
    
    # Preprocess the record
    X = preprocess_record(record, encoders, scaler, selected_features)
    
    # Make prediction
    prediction = model.predict(X)[0]
    
    # Get probability if available
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(X)[0]
        confidence = probabilities[prediction]
    else:
        confidence = None
    
    # Interpret result
    result = {
        'is_attack': bool(prediction),
        'label': 'ATTACK' if prediction == 1 else 'NORMAL',
        'confidence': confidence,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    return result

# DETECT FROM FILE
def detect_from_file(file_path, model, encoders, scaler, selected_features, output_file=None):
    """Detect attacks in a CSV file containing multiple connections"""
    
    print(f"\n📁 Processing file: {file_path}")
    
    try:
        # Load the file
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            df = pd.read_csv(file_path, names=COL_NAMES)
        
        print(f"   Found {len(df)} connections to analyze")
        
        # Process in batches to avoid memory issues
        results = []
        batch_size = 1000
        
        for i in range(0, len(df), batch_size):
            batch = df.iloc[i:i+batch_size]
            
            # Preprocess batch
            X_batch = preprocess_record(batch, encoders, scaler, selected_features)
            
            # Predict batch
            predictions = model.predict(X_batch)
            
            # Get probabilities if available
            if hasattr(model, 'predict_proba'):
                probs = model.predict_proba(X_batch)
                confidences = [probs[j][predictions[j]] for j in range(len(predictions))]
            else:
                confidences = [None] * len(predictions)
            
            # Store results
            for j, idx in enumerate(batch.index):
                results.append({
                    'index': idx,
                    'prediction': 'ATTACK' if predictions[j] == 1 else 'NORMAL',
                    'confidence': confidences[j],
                    'actual': batch.iloc[j].get('label', 'unknown') if 'label' in batch.columns else 'unknown'
                })
            
            print(f"   Progress: {min(i+batch_size, len(df))}/{len(df)} connections")
        
        # Convert to DataFrame
        results_df = pd.DataFrame(results)
        
        # Summary
        attacks = sum(results_df['prediction'] == 'ATTACK')
        normal = sum(results_df['prediction'] == 'NORMAL')
        
        print(f"\n📊 DETECTION SUMMARY")
        print(f"   Total connections: {len(results_df)}")
        print(f"   Attacks detected: {attacks} ({attacks/len(results_df)*100:.1f}%)")
        print(f"   Normal traffic: {normal} ({normal/len(results_df)*100:.1f}%)")
        
        # Save results if output file specified
        if output_file:
            results_df.to_csv(output_file, index=False)
            print(f"   ✅ Results saved to: {output_file}")
        
        return results_df
    
    except Exception as e:
        print(f"❌ Error processing file: {e}")
        return None

# INTERACTIVE MODE
def interactive_mode(model, encoders, scaler, selected_features):
    """Interactive mode where user enters connection details"""
    
    print("\n🖥️  INTERACTIVE DETECTION MODE")
    print("=" * 50)
    print("Enter connection details when prompted (or press Ctrl+C to exit)")
    print("(Press Enter to accept default values in brackets)")
    
    while True:
        print("\n" + "-" * 30)
        
        # Get user input
        try:
            # Basic fields for quick testing
            protocol = input("Protocol (tcp/udp/icmp) [tcp]: ").strip() or "tcp"
            service = input("Service (http/ftp/smtp/etc) [http]: ").strip() or "http"
            flag = input("Flag (SF/S0/REJ/etc) [SF]: ").strip() or "SF"
            src_bytes = input("Source bytes [100]: ").strip() or "100"
            dst_bytes = input("Destination bytes [200]: ").strip() or "200"
            duration = input("Duration (seconds) [0]: ").strip() or "0"
            
            # Create record with all required fields
            record = {
                'protocol_type': protocol,
                'service': service,
                'flag': flag,
                'src_bytes': int(src_bytes),
                'dst_bytes': int(dst_bytes),
                'duration': float(duration),
                # Default values for all other fields
                'land': 0,
                'wrong_fragment': 0,
                'urgent': 0,
                'hot': 0,
                'num_failed_logins': 0,
                'logged_in': 1,
                'num_compromised': 0,
                'root_shell': 0,
                'su_attempted': 0,
                'num_root': 0,
                'num_file_creations': 0,
                'num_shells': 0,
                'num_access_files': 0,
                'num_outbound_cmds': 0,
                'is_host_login': 0,
                'is_guest_login': 0,
                'count': 1,
                'srv_count': 1,
                'serror_rate': 0.0,
                'srv_serror_rate': 0.0,
                'rerror_rate': 0.0,
                'srv_rerror_rate': 0.0,
                'same_srv_rate': 1.0,
                'diff_srv_rate': 0.0,
                'srv_diff_host_rate': 0.0,
                'dst_host_count': 1,
                'dst_host_srv_count': 1,
                'dst_host_same_srv_rate': 1.0,
                'dst_host_diff_srv_rate': 0.0,
                'dst_host_same_src_port_rate': 0.0,
                'dst_host_srv_diff_host_rate': 0.0,
                'dst_host_serror_rate': 0.0,
                'dst_host_srv_serror_rate': 0.0,
                'dst_host_rerror_rate': 0.0,
                'dst_host_srv_rerror_rate': 0.0
            }
            
            # Detect
            result = detect_attack(record, model, encoders, scaler, selected_features)
            
            # Show result
            print(f"\n🔍 RESULT: {result['label']}")
            if result['confidence']:
                print(f"   Confidence: {result['confidence']*100:.1f}%")
            
        except KeyboardInterrupt:
            print("\n\nExiting interactive mode.")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            continue
        
        # Ask to continue
        again = input("\nAnalyze another? (y/n): ").strip().lower()
        if again != 'y':
            break

# SINGLE CONNECTION MODE
def single_connection_mode(protocol, service, flag, src_bytes, dst_bytes, model, encoders, scaler, selected_features):
    """Process a single connection from command line arguments"""
    
    record = {
        'protocol_type': protocol,
        'service': service,
        'flag': flag,
        'src_bytes': int(src_bytes),
        'dst_bytes': int(dst_bytes),
        'duration': 0,
        # Default values for all other fields
        'land': 0,
        'wrong_fragment': 0,
        'urgent': 0,
        'hot': 0,
        'num_failed_logins': 0,
        'logged_in': 1,
        'num_compromised': 0,
        'root_shell': 0,
        'su_attempted': 0,
        'num_root': 0,
        'num_file_creations': 0,
        'num_shells': 0,
        'num_access_files': 0,
        'num_outbound_cmds': 0,
        'is_host_login': 0,
        'is_guest_login': 0,
        'count': 1,
        'srv_count': 1,
        'serror_rate': 0.0,
        'srv_serror_rate': 0.0,
        'rerror_rate': 0.0,
        'srv_rerror_rate': 0.0,
        'same_srv_rate': 1.0,
        'diff_srv_rate': 0.0,
        'srv_diff_host_rate': 0.0,
        'dst_host_count': 1,
        'dst_host_srv_count': 1,
        'dst_host_same_srv_rate': 1.0,
        'dst_host_diff_srv_rate': 0.0,
        'dst_host_same_src_port_rate': 0.0,
        'dst_host_srv_diff_host_rate': 0.0,
        'dst_host_serror_rate': 0.0,
        'dst_host_srv_serror_rate': 0.0,
        'dst_host_rerror_rate': 0.0,
        'dst_host_srv_rerror_rate': 0.0
    }
    
    result = detect_attack(record, model, encoders, scaler, selected_features)
    
    print(f"\n🔍 RESULT: {result['label']}")
    if result['confidence']:
        print(f"   Confidence: {result['confidence']*100:.1f}%")
    
    return result

# MAIN FUNCTION
def main():
    parser = argparse.ArgumentParser(description='Network Intrusion Detection System')
    parser.add_argument('--file', '-f', type=str, help='CSV file to analyze')
    parser.add_argument('--output', '-o', type=str, help='Output file for results')
    parser.add_argument('--interactive', '-i', action='store_true', help='Interactive mode')
    parser.add_argument('--single', '-s', nargs=5, metavar=('PROTOCOL', 'SERVICE', 'FLAG', 'SRC_BYTES', 'DST_BYTES'),
                        help='Single connection: protocol service flag src_bytes dst_bytes')
    
    args = parser.parse_args()
    
    # Load model and artifacts
    model, encoders, scaler, selected_features = load_artifacts()
    if model is None:
        sys.exit(1)
    
    # Process based on arguments
    if args.file:
        # File mode
        detect_from_file(args.file, model, encoders, scaler, selected_features, args.output)
    
    elif args.single:
        # Single connection mode
        protocol, service, flag, src_bytes, dst_bytes = args.single
        single_connection_mode(protocol, service, flag, src_bytes, dst_bytes, 
                              model, encoders, scaler, selected_features)
    
    elif args.interactive:
        # Interactive mode
        interactive_mode(model, encoders, scaler, selected_features)
    
    else:
        # No arguments, show help
        parser.print_help()
        print("\n" + "="*50)
        print("📋 QUICK START EXAMPLES:")
        print("  python detect_attack.py --interactive")
        print("  python detect_attack.py --single tcp http SF 100 200")
        print("  python detect_attack.py --file sample_traffic.csv --output results.csv")

if __name__ == "__main__":
    main()