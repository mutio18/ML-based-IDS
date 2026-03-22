import pandas as pd
import joblib
import numpy as np

print("🔧 Loading model and data...")

# Load everything
model = joblib.load('models/best_model_latest.pkl')
X_train = pd.read_csv('data/processed/X_train_final.csv')
encoders = joblib.load('models/encoders.pkl')
scaler = joblib.load('models/scaler.pkl')

print(f"✅ Model loaded: {type(model).__name__}")
print(f"✅ Model expects {len(X_train.columns)} features")

print("\n" + "="*50)
print("🖥️  SIMPLE DETECTION MODE")
print("="*50)

while True:
    print("\n" + "-"*30)
    
    # Get input
    protocol = input("Protocol (tcp/udp/icmp) [tcp]: ").strip() or "tcp"
    service = input("Service (http/ftp/smtp/etc) [http]: ").strip() or "http"
    flag = input("Flag (SF/S0/REJ/etc) [SF]: ").strip() or "SF"
    src_bytes = input("Source bytes [100]: ").strip() or "100"
    dst_bytes = input("Destination bytes [200]: ").strip() or "200"
    duration = input("Duration (seconds) [0]: ").strip() or "0"
    
    # Create a copy of the first training row (all values are already scaled/encoded)
    test_row = X_train.iloc[[0]].copy()
    
    print("\n📊 Testing connection:")
    print(f"  Protocol: {protocol}, Service: {service}, Flag: {flag}")
    print(f"  Bytes: {src_bytes} → {dst_bytes}, Duration: {duration}s")
    
    # Create a raw input row with our values
    raw_data = pd.DataFrame([{
        'protocol_type': protocol,
        'service': service,
        'flag': flag,
        'src_bytes': int(src_bytes),
        'dst_bytes': int(dst_bytes),
        'duration': float(duration)
    }])
    
    # Encode categorical values
    for col in ['protocol_type', 'service', 'flag']:
        if col in encoders:
            le = encoders[col]
            raw_data[col] = raw_data[col].astype(str)
            raw_data[col] = raw_data[col].map(
                lambda x: le.transform([x])[0] if x in le.classes_ else -1
            )
    
    # Update the template row with our encoded values
    # Keep all other values from the template (they're already properly scaled)
    for col in raw_data.columns:
        if col in test_row.columns:
            test_row[col] = raw_data[col].values[0]
    
    # Make prediction
    pred = model.predict(test_row)[0]
    
    # Get probability
    prob = model.predict_proba(test_row)[0]
    confidence = prob[pred] * 100
    
    print(f"\n🔍 RESULT: {'🔴 ATTACK' if pred == 1 else '🟢 NORMAL'}")
    print(f"   Confidence: {confidence:.1f}%")
    print(f"   Probabilities: Normal={prob[0]*100:.1f}%, Attack={prob[1]*100:.1f}%")
    
    # Ask to continue
    again = input("\nTest another? (y/n): ").strip().lower()
    if again != 'y':
        break

print("\n✅ Done!")