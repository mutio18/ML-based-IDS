import pandas as pd
import joblib
import numpy as np

print("🔍 DEBUGGING MODEL INPUT")
print("="*50)

# Load everything
model = joblib.load('models/best_model_latest.pkl')
X_train = pd.read_csv('data/processed/X_train_final.csv')
encoders = joblib.load('models/encoders.pkl')
scaler = joblib.load('models/scaler.pkl')

print(f"Model type: {type(model).__name__}")
print(f"Training data shape: {X_train.shape}")
print(f"Training data columns: {list(X_train.columns)[:5]}...")

# Take one sample from training data (known normal)
normal_sample = X_train.iloc[0:1].copy()
print(f"\n✅ Using first row of training data as reference")
print(f"First row values (first 5): {normal_sample.iloc[0, :5].tolist()}")

# Predict on this training sample
pred = model.predict(normal_sample)[0]
if hasattr(model, 'predict_proba'):
    prob = model.predict_proba(normal_sample)[0]
    print(f"\n📊 Model prediction on TRAINING data:")
    print(f"   Prediction: {'ATTACK' if pred == 1 else 'NORMAL'}")
    print(f"   Confidence: {prob[pred]*100:.1f}%")
    print(f"   Probabilities: Normal={prob[0]*100:.1f}%, Attack={prob[1]*100:.1f}%")

# Now test with a simple HTTP connection using same preprocessing
print("\n" + "="*50)
print("Testing HTTP connection with proper preprocessing:")

# Get original feature columns (before engineering)
original_cols = [c for c in X_train.columns if c not in ['bytes_ratio', 'error_ratio', 'total_bytes']]

# Create a simple test row with original features
test_dict = {
    'duration': 0,
    'protocol_type': 'tcp',
    'service': 'http',
    'flag': 'SF',
    'src_bytes': 217,
    'dst_bytes': 2032,
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

test_df = pd.DataFrame([test_dict])

# Apply encoding
for col, le in encoders.items():
    if col in test_df.columns:
        test_df[col] = test_df[col].astype(str)
        test_df[col] = test_df[col].map(
            lambda x: le.transform([x])[0] if x in le.classes_ else -1
        )

# Create engineered features
test_df['bytes_ratio'] = test_df['src_bytes'] / (test_df['dst_bytes'] + 1)
test_df['total_bytes'] = test_df['src_bytes'] + test_df['dst_bytes']
test_df['error_ratio'] = 0

# Apply scaling (to all columns)
num_cols = test_df.columns.tolist()
test_df[num_cols] = scaler.transform(test_df[num_cols])

# Ensure correct columns
test_df = test_df[X_train.columns]

# Predict
pred = model.predict(test_df)[0]
if hasattr(model, 'predict_proba'):
    prob = model.predict_proba(test_df)[0]
    print(f"\n📊 Model prediction on HTTP connection:")
    print(f"   Prediction: {'ATTACK' if pred == 1 else 'NORMAL'}")
    print(f"   Confidence: {prob[pred]*100:.1f}%")
    print(f"   Probabilities: Normal={prob[0]*100:.1f}%, Attack={prob[1]*100:.1f}%")

print("\n✅ Debug complete!")