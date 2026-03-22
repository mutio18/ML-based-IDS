import os
import pandas as pd

print("Current directory:", os.getcwd())

print("\nFiles in data/raw:")
for file in os.listdir("data/raw"):
    print("  -", file)

print("\nTrying to load KDDTrain+.txt...")
df = pd.read_csv("data/raw/KDDTrain+.txt", header=None, nrows=5)
print("SUCCESS! DataFrame shape:", df.shape)
print("\nFirst 5 rows (first 5 columns):")
print(df.iloc[:5, :5])