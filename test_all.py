import subprocess
import os

test_files = [
    'data/UNSW/UNSW_NB15_testing-set.csv',
    'ddos_test.csv',
    'port_scan.csv',
    'normal_traffic.csv'
]

for test_file in test_files:
    if os.path.exists(test_file):
        print(f"\n{'='*60}")
        print(f"Testing: {test_file}")
        print('='*60)
        output = f"results_{os.path.basename(test_file)}"
        subprocess.run([
            'python', 'production_ids.py',
            '--file', test_file,
            '--output', output
        ])