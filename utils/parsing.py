import argparse
import yaml
import torch

def parse_basic_args():
    """Parse basic command-line arguments"""
    parser = argparse.ArgumentParser(description='River segmentation/classification training script')
    
    # Basic arguments
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--task', type=str, default='segmentation', 
                        choices=['segmentation', 'classification', 'combined'], help='Task type')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    
    # Allow overriding any config value with key-value pairs
    parser.add_argument('--set', nargs=2, action='append', metavar=('key', 'value'), 
                        help='Override config values: --set key value (can be used multiple times)')
    
    return parser.parse_args()

def update_nested_dict(d, key_path, value):
    """Update a nested dictionary using a dot-separated key path."""
    keys = key_path.split('.')
    for key in keys[:-1]:
        if key not in d:
            d[key] = {}
        d = d[key]
    
    # Try to convert the value to the right type
    try:
        # Try to convert to int
        value = int(value)
    except ValueError:
        try:
            # Try to convert to float
            value = float(value)
        except ValueError:
            # It's a string or boolean
            if value.lower() == 'true':
                value = True
            elif value.lower() == 'false':
                value = False
            # else leave it as string
    
    d[keys[-1]] = value

def load_and_process_config(args):
    """Load config from file and update with command-line arguments"""
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override config with command line arguments
    if args.set:
        for key_path, value in args.set:
            update_nested_dict(config, key_path, value)
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    
    return config, device