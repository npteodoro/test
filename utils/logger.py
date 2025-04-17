import os
import time

def setup_logger(config, args, metrics):
    """Create a log entry for the experiment with all details"""
    log_dir = "experiments"
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    # Create task-specific log files
    if args.task == "segmentation":
        log_file = f"{log_dir}/segmentation_experiments.csv"
        
        # Create headers if file doesn't exist
        if not os.path.exists(log_file):
            with open(log_file, 'w') as f:
                headers = ["timestamp", "encoder", "decoder", "augmentation", 
                           "batch_size", "learning_rate", "dice", "iou"]
                f.write(",".join(headers) + "\n")
        
        # Add entry to log
        with open(log_file, 'a') as f:
            values = [
                timestamp,
                config['model'].get('encoder', 'default'),
                config['model'].get('decoder', 'default'),
                config['data'].get('augmentation', 'default'),
                str(config['training']['batch_size']),
                str(config['training']['learning_rate']),
                str(metrics.get('dice', 'N/A')), 
                str(metrics.get('iou', 'N/A'))
            ]
            f.write(",".join(values) + "\n")
    
    elif args.task == "classification":
        log_file = f"{log_dir}/classification_experiments.csv"
        
        # Create headers if file doesn't exist
        if not os.path.exists(log_file):
            with open(log_file, 'w') as f:
                headers = ["timestamp", "encoder", "use_mask_as_channel", "augmentation", 
                           "batch_size", "learning_rate", "accuracy", "f1", "precision", "recall"]
                f.write(",".join(headers) + "\n")
        
        # Add entry to log
        with open(log_file, 'a') as f:
            values = [
                timestamp,
                config['model'].get('encoder', 'default'),
                str(config['model'].get('use_mask_as_channel', False)),
                config['data'].get('augmentation', 'default'),
                str(config['training']['batch_size']),
                str(config['training']['learning_rate']),
                str(metrics.get('accuracy', 'N/A')), 
                str(metrics.get('f1', 'N/A')),
                str(metrics.get('precision', 'N/A')), 
                str(metrics.get('recall', 'N/A'))
            ]
            f.write(",".join(values) + "\n")