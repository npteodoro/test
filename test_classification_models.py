#!/usr/bin/env python3
# test_classification_models.py

import subprocess
import os
import time
import pandas as pd
from datetime import datetime
import re
import itertools
import yaml

# Configuration
EPOCHS = 25
OUTPUT_DIR = f"results/classification_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Define the combinations to test
encoders = ["efficientnet_b2", "mobilenet_v3_large", "squeezenet1_1", "shufflenet_v2_x1_0", "mnasnet1_0"]
augmentations = ["base", "combined", "geometric", "semantic"]
mask_methods = [None, "channel", "attention", "feature_weighting", "region_focus", "attention_layer"]

# Create a log file to track all tests
log_file = os.path.join(OUTPUT_DIR, "testing_log.csv")
with open(log_file, "w") as f:
    f.write("encoder,mask_method,augmentation,status,accuracy,f1,precision,recall,training_time\n")

# Create all combinations
combinations = list(itertools.product(encoders, mask_methods, augmentations))

# Run each combination
for i, (encoder, mask_method, augmentation) in enumerate(combinations):
    # Create readable names
    mask_name = "none" if mask_method is None else mask_method
    test_name = f"{encoder}_{mask_name}_{augmentation}"
    output_file = os.path.join(OUTPUT_DIR, f"{test_name}.log")
    
    print(f"\n[{i+1}/{len(combinations)}] Testing: {test_name}")
    print(f"  - Encoder: {encoder}")
    print(f"  - Mask Method: {mask_name}")
    print(f"  - Augmentation: {augmentation}")
    
    # Build command
    cmd = ["python", "main.py", "--task", "classification"]
    cmd.extend(["--set", "model.encoder", encoder])
    cmd.extend(["--set", "data.augmentation", augmentation])
    cmd.extend(["--set", "training.num_epochs", str(EPOCHS)])
    cmd.extend(["--set", "training.save_checkpoints", "false"])
    
    # Handle different mask methods
    if mask_method is not None:
        cmd.extend(["--set", "model.mask_method", mask_method])
        cmd.extend(["--set", "data.mask_method", mask_method])
        
        # For some mask methods, we might need additional settings
        if mask_method == "feature_weighting" or mask_method == "attention":
            cmd.extend(["--set", "data.mask_weight", "0.5"])
    
    # Run the command and capture output
    start_time = time.time()
    with open(output_file, "w") as f:
        process = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    
    training_time = time.time() - start_time
    
    # Extract metrics from the log file
    metrics = {"accuracy": "N/A", "f1": "N/A", "precision": "N/A", "recall": "N/A"}
    status = "SUCCESS" if process.returncode == 0 else "FAILED"
    
    if process.returncode == 0:
        with open(output_file, "r") as f:
            log_content = f.read()
            
            # Extract metrics from the log
            accuracy_match = re.search(r"Test Metrics - Accuracy: ([\d\.]+)", log_content)
            f1_match = re.search(r"F1: ([\d\.]+)", log_content)
            precision_match = re.search(r"Precision: ([\d\.]+)", log_content)
            recall_match = re.search(r"Recall: ([\d\.]+)", log_content)
            
            if accuracy_match:
                metrics["accuracy"] = accuracy_match.group(1)
            if f1_match:
                metrics["f1"] = f1_match.group(1)
            if precision_match:
                metrics["precision"] = precision_match.group(1)
            if recall_match:
                metrics["recall"] = recall_match.group(1)
    
    # Log results to CSV
    with open(log_file, "a") as f:
        f.write(f"{encoder},{mask_name},{augmentation},{status},{metrics['accuracy']},"
                f"{metrics['f1']},{metrics['precision']},{metrics['recall']},{training_time:.1f}\n")
    
    # Print results summary
    print(f"  Status: {status}")
    print(f"  Time: {training_time:.1f} seconds")
    print(f"  Metrics: Accuracy={metrics['accuracy']}, F1={metrics['f1']}")

# Generate an HTML report at the end
df = pd.read_csv(log_file)

# Create an HTML report with result tables
html_report = f"""
<html>
<head>
    <title>Classification Testing Results</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        tr:nth-child(even) {{ background-color: #f9f9f9; }}
        .best {{ background-color: #d4edda; font-weight: bold; }}
        h1, h2 {{ color: #333; }}
    </style>
</head>
<body>
    <h1>Classification Model Testing Results</h1>
    <p>Testing completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    
    <h2>All Results</h2>
    {df.to_html(index=False)}
    
    <h2>Best Model by Accuracy</h2>
    {df.loc[df['accuracy'].astype(float).idxmax() if 'N/A' not in df['accuracy'].values else 0].to_frame().T.to_html(index=False)}
    
    <h2>Best Model by F1 Score</h2>
    {df.loc[df['f1'].astype(float).idxmax() if 'N/A' not in df['f1'].values else 0].to_frame().T.to_html(index=False)}
</body>
</html>
"""

with open(os.path.join(OUTPUT_DIR, "report.html"), "w") as f:
    f.write(html_report)

print(f"\nTesting completed. Results saved to {OUTPUT_DIR}")
print(f"Summary report available at {os.path.join(OUTPUT_DIR, 'report.html')}")

# Create best model configuration YAML


best_acc_idx = df['accuracy'].astype(float).idxmax() if 'N/A' not in df['accuracy'].values else 0
best_model = df.iloc[best_acc_idx]

best_config = {
    'classification': {
        'encoder': best_model['encoder'],
        'mask_method': None if best_model['mask_method'] == 'none' else best_model['mask_method'],
        'augmentation': best_model['augmentation'],
        'metrics': {
            'accuracy': float(best_model['accuracy']),
            'f1': float(best_model['f1']),
            'precision': float(best_model['precision']),
            'recall': float(best_model['recall'])
        }
    }
}

with open(os.path.join(OUTPUT_DIR, "best_config.yaml"), "w") as f:
    yaml.dump(best_config, f, default_flow_style=False)

print(f"Best configuration saved to {os.path.join(OUTPUT_DIR, 'best_config.yaml')}")