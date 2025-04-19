#!/usr/bin/env python3
# test_segmentation_models.py

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
OUTPUT_DIR = f"results/segmentation_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Define the combinations to test
encoders = ["efficientnet_b2", "mobilenet_v3_large", "squeezenet1_1", "shufflenet_v2_x1_0", "mnasnet1_0"]
decoders = ["unet", "fpn", "deeplabv3"]
augmentations = ["base", "combined", "geometric", "semantic"]

# Create a log file to track all tests
log_file = os.path.join(OUTPUT_DIR, "testing_log.csv")
with open(log_file, "w") as f:
    f.write("encoder,decoder,augmentation,status,dice,iou,training_time\n")

# Create all combinations
combinations = list(itertools.product(encoders, decoders, augmentations))

# Run each combination
for i, (encoder, decoder, augmentation) in enumerate(combinations):
    test_name = f"{encoder}_{decoder}_{augmentation}"
    output_file = os.path.join(OUTPUT_DIR, f"{test_name}.log")
    
    print(f"\n[{i+1}/{len(combinations)}] Testing: {test_name}")
    print(f"  - Encoder: {encoder}")
    print(f"  - Decoder: {decoder}")
    print(f"  - Augmentation: {augmentation}")
    
    # Build command
    cmd = ["python", "main.py", "--task", "segmentation"]
    cmd.extend(["--set", "model.encoder", encoder])
    cmd.extend(["--set", "model.segmentation.decoder", decoder])
    cmd.extend(["--set", "data.augmentation", augmentation])
    cmd.extend(["--set", "training.num_epochs", str(EPOCHS)])
    cmd.extend(["--set", "training.save_checkpoints", "false"])
    
    # Run the command and capture output
    start_time = time.time()
    with open(output_file, "w") as f:
        process = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    
    training_time = time.time() - start_time
    
    # Extract metrics from the log file
    metrics = {"dice": "N/A", "iou": "N/A"}
    status = "SUCCESS" if process.returncode == 0 else "FAILED"
    
    if process.returncode == 0:
        with open(output_file, "r") as f:
            log_content = f.read()
            
            # Extract metrics from the log
            dice_match = re.search(r"Test Metrics - Dice: ([\d\.]+)", log_content)
            iou_match = re.search(r"IoU: ([\d\.]+)", log_content)
            
            if dice_match:
                metrics["dice"] = dice_match.group(1)
            if iou_match:
                metrics["iou"] = iou_match.group(1)
    
    # Log results to CSV
    with open(log_file, "a") as f:
        f.write(f"{encoder},{decoder},{augmentation},{status},{metrics['dice']},"
                f"{metrics['iou']},{training_time:.1f}\n")
    
    # Print results summary
    print(f"  Status: {status}")
    print(f"  Time: {training_time:.1f} seconds")
    print(f"  Metrics: Dice={metrics['dice']}, IoU={metrics['iou']}")

# Generate an HTML report at the end
df = pd.read_csv(log_file)

# Filter out failed runs
df_valid = df[df['status'] == 'SUCCESS'].copy()
df_valid['dice'] = pd.to_numeric(df_valid['dice'], errors='coerce')
df_valid['iou'] = pd.to_numeric(df_valid['iou'], errors='coerce')

# Create an HTML report with result tables
html_report = f"""
<html>
<head>
    <title>Segmentation Testing Results</title>
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
    <h1>Segmentation Model Testing Results</h1>
    <p>Testing completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    
    <h2>All Results</h2>
    {df.to_html(index=False)}
"""

# Add best model sections only if we have valid results
if len(df_valid) > 0:
    best_dice_idx = df_valid['dice'].idxmax()
    best_iou_idx = df_valid['iou'].idxmax()
    
    html_report += f"""
    <h2>Best Model by Dice Score</h2>
    {df.loc[best_dice_idx].to_frame().T.to_html(index=False)}
    
    <h2>Best Model by IoU Score</h2>
    {df.loc[best_iou_idx].to_frame().T.to_html(index=False)}
    """

html_report += """
</body>
</html>
"""

with open(os.path.join(OUTPUT_DIR, "report.html"), "w") as f:
    f.write(html_report)

print(f"\nTesting completed. Results saved to {OUTPUT_DIR}")
print(f"Summary report available at {os.path.join(OUTPUT_DIR, 'report.html')}")

# Create best model configuration YAML


if len(df_valid) > 0:
    best_dice_idx = df_valid['dice'].idxmax()
    best_model = df.loc[best_dice_idx]

    best_config = {
        'segmentation': {
            'encoder': best_model['encoder'],
            'decoder': best_model['decoder'],
            'augmentation': best_model['augmentation'],
            'metrics': {
                'dice': float(best_model['dice']),
                'iou': float(best_model['iou']),
            }
        }
    }

    with open(os.path.join(OUTPUT_DIR, "best_config.yaml"), "w") as f:
        yaml.dump(best_config, f, default_flow_style=False)

    print(f"Best configuration saved to {os.path.join(OUTPUT_DIR, 'best_config.yaml')}")
else:
    print("No successful runs to determine best configuration")