# analyze_attack_types.py
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

print("="*60)
print(" ATTACK TYPE ANALYSIS")
print("="*60)

# Load test data with attack categories
test_raw = pd.read_parquet('data/UNSW/UNSW_NB15_testing-set.parquet')
print(f"Test set size: {len(test_raw)}")

# Load your predictions
results = pd.read_csv('unsw_results.csv')
print(f"Predictions loaded: {len(results)}")

# Add predictions to test data
test_raw['prediction'] = results['prediction'].values
test_raw['pred_label'] = (test_raw['prediction'] == 'ATTACK').astype(int)

# ============================================
# 1. Overall Attack Type Distribution
# ============================================
print("\n ATTACK TYPE DISTRIBUTION IN TEST SET:")
attack_dist = test_raw['attack_cat'].value_counts()
for cat, count in attack_dist.items():
    pct = count/len(test_raw)*100
    print(f"   {cat:15s}: {count:6d} ({pct:5.2f}%)")

# ============================================
# 2. Performance by Attack Type
# ============================================
print("\n DETECTION RATE BY ATTACK TYPE:")
print("-" * 60)
print(f"{'Attack Type':<20s} {'Total':<8s} {'Detected':<10s} {'Rate':<8s} {'Missed':<8s}")
print("-" * 60)

attack_types = test_raw[test_raw['label'] == 1]['attack_cat'].unique()
results_by_type = []

for attack in attack_types:
    # Get all instances of this attack
    attack_instances = test_raw[test_raw['attack_cat'] == attack]
    total = len(attack_instances)
    
    if total > 0:
        detected = attack_instances['pred_label'].sum()
        rate = detected / total * 100
        missed = total - detected
        
        print(f"{attack:<20s} {total:<8d} {detected:<10d} {rate:<8.2f}% {missed:<8d}")
        
        results_by_type.append({
            'attack_type': attack,
            'total': total,
            'detected': detected,
            'detection_rate': rate,
            'missed': missed
        })

# ============================================
# 3. Normal Traffic Performance
# ============================================
normal = test_raw[test_raw['label'] == 0]
normal_total = len(normal)
normal_correct = (normal['pred_label'] == 0).sum()
normal_false_alarms = normal_total - normal_correct

print("\n NORMAL TRAFFIC PERFORMANCE:")
print(f"   Total Normal:        {normal_total:6d}")
print(f"   Correctly Identified: {normal_correct:6d} ({normal_correct/normal_total*100:.2f}%)")
print(f"   False Alarms:        {normal_false_alarms:6d} ({normal_false_alarms/normal_total*100:.2f}%)")

# ============================================
# 4. Summary Table
# ============================================
print("\n ATTACK TYPE SUMMARY:")
print("-" * 60)
print(f"{'Attack Type':<20s} {'Total':<8s} {'Detected':<8s} {'Rate':<8s} {'Missed':<8s}")
print("-" * 60)

results_df = pd.DataFrame(results_by_type)
results_df = results_df.sort_values('detection_rate', ascending=False)

for _, row in results_df.iterrows():
    print(f"{row['attack_type']:<20s} {row['total']:<8d} {row['detected']:<8d} {row['detection_rate']:<8.2f}% {row['missed']:<8d}")

# ============================================
# 5. Visualization
# ============================================
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))

# Detection rates bar chart
plt.subplot(1, 2, 1)
colors = ['green' if r >= 95 else 'orange' if r >= 85 else 'red' for r in results_df['detection_rate']]
plt.barh(results_df['attack_type'], results_df['detection_rate'], color=colors)
plt.xlabel('Detection Rate (%)')
plt.title('Detection Rate by Attack Type')
plt.xlim(0, 100)
plt.axvline(x=90, color='blue', linestyle='--', alpha=0.5, label='90% Target')
plt.legend()

# Missed attacks count
plt.subplot(1, 2, 2)
plt.barh(results_df['attack_type'], results_df['missed'], color='red', alpha=0.7)
plt.xlabel('Number Missed')
plt.title('Missed Attacks by Type')

plt.tight_layout()
plt.savefig('results/UNSW/attack_type_performance.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================
# 6. Save Results
# ============================================
results_df.to_csv('results/UNSW/attack_type_analysis.csv', index=False)
print("\n Attack type analysis saved to: results/UNSW/attack_type_analysis.csv")

# ============================================
# 7. Best and Worst Performing Attacks
# ============================================
print("\n BEST DETECTED ATTACKS:")
best = results_df.nlargest(3, 'detection_rate')
for _, row in best.iterrows():
    print(f"   {row['attack_type']}: {row['detection_rate']:.2f}% ({row['detected']}/{row['total']})")

print("\n WORST DETECTED ATTACKS (needs improvement):")
worst = results_df.nsmallest(3, 'detection_rate')
for _, row in worst.iterrows():
    print(f"   {row['attack_type']}: {row['detection_rate']:.2f}% ({row['detected']}/{row['total']})")