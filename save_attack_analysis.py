# save_attack_analysis.py
import pandas as pd

# Load the attack type analysis
print("Loading attack type analysis...")
results_df = pd.read_csv('results/UNSW/attack_type_analysis.csv')

# Save a final copy
results_df.to_csv('results/UNSW/attack_type_final.csv', index=False)
print(" Attack type analysis saved to: results/UNSW/attack_type_final.csv")

# Display summary
print("\n ATTACK TYPE PERFORMANCE SUMMARY:")
print(results_df.to_string(index=False))

# Calculate overall metrics
avg_detection = results_df['detection_rate'].mean()
weighted_detection = (results_df['detected'].sum() / results_df['total'].sum()) * 100

print(f"\n Overall Statistics:")
print(f"   Average Detection Rate: {avg_detection:.2f}%")
print(f"   Weighted Detection Rate: {weighted_detection:.2f}%")
print(f"   Total Attacks: {results_df['total'].sum():,}")
print(f"   Total Detected: {results_df['detected'].sum():,}")