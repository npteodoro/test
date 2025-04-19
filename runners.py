import torch
import os
import time
from factories.model import create_model
from factories.trainer import create_trainer
from factories.evaluator import create_evaluator
from utils.logger import setup_logger

def run_experiment(args, config, device):
    """Run a complete training and evaluation experiment"""
    start_time = time.time()
    
    try:
        # Create model
        model = create_model(args.task, config['model'])
        
        # Create criterion
        if args.task == 'segmentation':
            criterion = torch.nn.BCEWithLogitsLoss()
        else:  # classification
            criterion = torch.nn.CrossEntropyLoss()
        
        # Create optimizer
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training'].get('weight_decay', 0)
        )
        
        # Add AMP (Automatic Mixed Precision) for faster training
        if device.type == 'cuda':
            # Add to config so trainer can use it
            config['training']['use_amp'] = config['training'].get('use_amp', True)
        
        # Create trainer
        trainer = create_trainer(
            args.task,
            model,
            criterion,
            optimizer,
            device,
            config['training']
        )
        
        # Train model
        model, history = trainer.train(
            args.train_loader,
            args.val_loader,
            config['training']['num_epochs']
        )
        
        # Print training time
        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
        
        # Save final model
        save_model(model, args, config)
        
        # Prepare for evaluation
        prepare_evaluation_config(args, config)
        
        # Evaluate on test set
        evaluator = create_evaluator(args.task, model, device, config.get('evaluation', {}))
        metrics = evaluator.evaluate(args.test_loader)
        
        # Log results
        setup_logger(config, args, metrics)
        
        if config.get('save_results', False):
            evaluator.save_results(metrics)
        
        return metrics
    
    except Exception as e:
        print(f"Error during experiment: {e}")
        import traceback
        traceback.print_exc()
        return None
    finally:
        # Clean up
        if device.type == 'cuda':
            torch.cuda.empty_cache()

def save_model(model, args, config):
    """Save the trained model"""
    model_dir = f"models/{args.task}"
    os.makedirs(model_dir, exist_ok=True)

    encoder_type = config['model'].get('encoder', 'default')
    
    # Handle different model types with different naming conventions
    if args.task == 'segmentation':
        decoder_type = config['model'].get('decoder', 'default')
        model_part = f"{encoder_type}_{decoder_type}"
    elif args.task == 'classification':
        mask_method = config['model'].get('mask_method', 'none')
        mask_str = f"_{mask_method}" if mask_method else ""
        model_part = f"{encoder_type}{mask_str}"
    elif args.task == 'combined':
        decoder_type = config['model'].get('decoder', 'default')
        mask_method = config['model'].get('mask_method', 'channel')
        model_part = f"{encoder_type}_{decoder_type}_{mask_method}"
    else:
        model_part = encoder_type
    
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    model_filename = f"{model_part}_{timestamp}.pth"
    
    torch.save(model.state_dict(), os.path.join(model_dir, model_filename))
    print(f"Model saved to {os.path.join(model_dir, model_filename)}")

def prepare_evaluation_config(args, config):
    """Prepare evaluation configuration"""
    if 'evaluation' not in config:
        config['evaluation'] = {}
    
    config['evaluation']['model_name'] = (
        f"{args.task}_{config['model'].get('encoder', 'default')}"
        f"{('_' + config['model'].get('decoder', '')) if args.task == 'segmentation' else ''}"
    )