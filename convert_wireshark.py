import pandas as pd
import os

print("="*60)
print(" WIRESHARK TO IDS CONVERTER")
print("="*60)

# Ask for file name
file_name = input("Enter your Wireshark CSV file name: ").strip()

if not file_name.endswith('.csv'):
    file_name = file_name + '.csv'

# Check if file exists
if not os.path.exists(file_name):
    print(f" File '{file_name}' not found!")
    print("Files in current folder:")
    for f in os.listdir('.'):
        if f.endswith('.csv'):
            print(f"   - {f}")
    exit(1)

# Load the file
print(f" Loading {file_name}...")
wireshark_df = pd.read_csv(file_name)
print(f" Loaded {len(wireshark_df)} packets")

# Print columns
print(f"\n Columns: {list(wireshark_df.columns)}")

# Convert to IDS format
print("\n Converting...")
converted = []
for idx, row in wireshark_df.iterrows():
    proto = str(row.get('Protocol', 'tcp')).lower()
    length = row.get('Length', 500)
    
    converted.append({
        'proto': proto,
        'service': 'unknown',
        'sbytes': length,
        'dbytes': length,
        'rate': 10,
        'dur': 1
    })

# Save
ids_df = pd.DataFrame(converted)
ids_df.to_csv('ids_input.csv', index=False)

print(f"\n Created ids_input.csv with {len(ids_df)} connections")