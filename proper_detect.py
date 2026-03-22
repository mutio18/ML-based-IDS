import pandas as pd
import joblib
import numpy as np

print("🔧 Loading model and data...")

# Load everything
model = joblib.load('models/best_model_latest.pkl')
X_train = pd.read_csv('data/processed/X_train_final.csv')
encoders = joblib.load('models/encoders.pkl')
scaler = joblib.load('models/scaler.pkl')

# Get the feature names from the scaler (this is the critical fix!)
if hasattr(scaler, 'feature_names_in_'):
    original_features = list(scaler.feature_names_in_)
    print(f" Got {len(original_features)} feature names from scaler")
else:
    # Fallback if scaler doesn't have feature names
    original_features = [
        'same_srv_rate', 'src_bytes', 'dst_host_srv_count', 'flag',
        'dst_host_same_srv_rate', 'dst_host_srv_serror_rate', 'dst_host_serror_rate',
        'dst_host_count', 'logged_in', 'dst_host_same_src_port_rate', 'count',
        'srv_count', 'srv_serror_rate', 'dst_host_srv_diff_host_rate', 'serror_rate',
        'dst_host_diff_srv_rate', 'protocol_type', 'service', 'duration',
        'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins',
        'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
        'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
        'is_guest_login', 'rerror_rate', 'srv_rerror_rate', 'diff_srv_rate',
        'srv_diff_host_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate'
    ]

# The 18 features your model uses after engineering
final_features = list(X_train.columns)
print(f" Model expects these {len(final_features)} final features")

print("\n" + "="*60)
print("  PROPER DETECTION MODE")
print("="*60)

while True:
    print("\n" + "-"*40)
    
    # Get basic input
    protocol = input("Protocol (tcp/udp/icmp) [tcp]: ").strip() or "tcp"
    service = input("Service (http/ftp/smtp/dns) [http]: ").strip() or "http"
    flag = input("Flag (SF/S0/REJ) [SF]: ").strip() or "SF"
    src_bytes = int(input("Source bytes [100]: ").strip() or "100")
    dst_bytes = int(input("Destination bytes [200]: ").strip() or "200")
    duration = float(input("Duration (seconds) [0]: ").strip() or "0")
    
    print("\n Creating full feature set...")
    
    # Create a dictionary with ALL original features
    feature_dict = {
        # Basic features from input
        'protocol_type': protocol,
        'service': service,
        'flag': flag,
        'src_bytes': src_bytes,
        'dst_bytes': dst_bytes,
        'duration': duration,
        
        # Default values for all other features
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
        'same_srv_rate': 1.0 if flag == 'SF' else 0.0,
        'diff_srv_rate': 0.0,
        'srv_diff_host_rate': 0.0,
        'dst_host_count': 1,
        'dst_host_srv_count': 1,
        'dst_host_same_srv_rate': 1.0 if flag == 'SF' else 0.0,
        'dst_host_diff_srv_rate': 0.0,
        'dst_host_same_src_port_rate': 0.0,
        'dst_host_srv_diff_host_rate': 0.0,
        'dst_host_serror_rate': 0.0,
        'dst_host_srv_serror_rate': 0.0,
        'dst_host_rerror_rate': 0.0,
        'dst_host_srv_rerror_rate': 0.0
    }
    
    # Convert to DataFrame
    test_df = pd.DataFrame([feature_dict])
    
    # Encode categorical features
    for col in ['protocol_type', 'service', 'flag']:
        if col in test_df.columns and col in encoders:
            le = encoders[col]
            test_df[col] = test_df[col].astype(str)
            test_df[col] = test_df[col].map(
                lambda x: le.transform([x])[0] if x in le.classes_ else -1
            )
            print(f"   {col} encoded to: {test_df[col].values[0]}")
    
    # Ensure all original features exist
    for col in original_features:
        if col not in test_df.columns:
            test_df[col] = 0
    
    # Apply scaling - using EXACTLY the feature order from the scaler
    test_df_scaled = test_df[original_features].copy()
    test_df_scaled = pd.DataFrame(
        scaler.transform(test_df_scaled),
        columns=original_features  # Keep the same column names
    )
    
    # Add engineered features
    test_df_scaled['bytes_ratio'] = src_bytes / (dst_bytes + 1)
    test_df_scaled['total_bytes'] = src_bytes + dst_bytes
    test_df_scaled['error_ratio'] = 0.0
    
    # Select only the features the model expects
    test_df_final = test_df_scaled[final_features]
    
    # Predict
    pred = model.predict(test_df_final)[0]
    prob = model.predict_proba(test_df_final)[0]
    
    print(f"\n🔍 RESULT: {'🔴 ATTACK' if pred == 1 else '🟢 NORMAL'}")
    print(f"   Confidence: {prob[pred]*100:.1f}%")
    print(f"   Probabilities: Normal={prob[0]*100:.1f}%, Attack={prob[1]*100:.1f}%")
    
    again = input("\nTest another? (y/n): ").strip().lower()
    if again != 'y':
        break

print("\n Done!")