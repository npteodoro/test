import torch
from factories.dataset import create_segmentation_datasets, create_classification_datasets
from utils.logger import setup_logger
from utils.config import validate_and_update_config
from utils.parsing import parse_basic_args, load_and_process_config
from utils.data import create_data_loader
from runners import run_experiment
import yaml
import traceback
import time

def main():
    # Parse arguments and load config
    args = parse_basic_args()
    config, device = load_and_process_config(args)
    
    # Validate and update config based on task
    config = validate_and_update_config(config, args.task)
    
    # Print config for user verification
    print(f"Running task: {args.task} with configuration:")
    print(yaml.dump(config, default_flow_style=False))
    
    # Start timing
    start_time = time.time()
    print(f"Starting {args.task} task at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check for required configuration sections
    required_keys = ['data', 'model', 'training']
    missing_keys = [key for key in required_keys if key not in config]
    if missing_keys:
        print(f"Error: Missing required configuration sections: {missing_keys}")
        return
    
    try:
        # Create datasets
        if args.task == 'segmentation':
            train_dataset, val_dataset, test_dataset = create_segmentation_datasets(
                csv_file=config['data']['csv_file'],
                root_dir=config['data']['root_dir'],
                config=config['data']
            )
        else:  # classification
            use_mask = config['data'].get('use_mask_as_channel', False)
            train_dataset, val_dataset, test_dataset = create_classification_datasets(
                csv_file=config['data']['csv_file'],
                root_dir=config['data']['root_dir'],
                config=config['data'],
                use_mask=use_mask
            )
            
        # Create data loaders
        args.train_loader = create_data_loader(train_dataset, config, shuffle=True)
        args.val_loader = create_data_loader(val_dataset, config)
        args.test_loader = create_data_loader(test_dataset, config)
        
        # Run the experiment
        metrics = run_experiment(args, config, device)
        
        # Print total execution time
        total_time = time.time() - start_time
        print(f"Total execution time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
        
    except FileNotFoundError as e:
        print(f"Error: Could not find required file: {e}")
    except Exception as e:
        print(f"Error during execution: {e}")
        traceback.print_exc()

if __name__ == '__main__':
    main()