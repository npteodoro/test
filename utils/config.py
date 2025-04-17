def validate_and_update_config(config, task_type):
    """
    Validate and update configuration based on task type
    Ensures that appropriate metrics and parameters are set
    """
    if 'training' not in config:
        config['training'] = {}

    # Set task-specific defaults
    if task_type == 'segmentation':
        # Segmentation metrics
        if config['training'].get('monitor_metric') not in ['dice', 'iou', 'loss']:
            print(f"Warning: '{config['training'].get('monitor_metric')}' is not a valid segmentation metric.")
            print("Setting monitor_metric to 'dice'")
            config['training']['monitor_metric'] = 'dice'
        
        if config['training'].get('monitor_mode') != 'max' and config['training']['monitor_metric'] != 'loss':
            print(f"Warning: For segmentation metrics dice/iou, mode should be 'max'")
            config['training']['monitor_mode'] = 'max'
            
    elif task_type == 'classification':
        # Classification metrics
        valid_metrics = ['accuracy', 'f1', 'precision', 'recall', 'loss']
        if config['training'].get('monitor_metric') not in valid_metrics:
            print(f"Warning: '{config['training'].get('monitor_metric')}' is not a valid classification metric.")
            print("Setting monitor_metric to 'accuracy'")
            config['training']['monitor_metric'] = 'accuracy'
        
        if config['training'].get('monitor_mode') != 'max' and config['training']['monitor_metric'] != 'loss':
            print(f"Warning: For classification metrics accuracy/f1/precision/recall, mode should be 'max'")
            config['training']['monitor_mode'] = 'max'
    
    # Ensure evaluation config exists
    if 'evaluation' not in config:
        config['evaluation'] = {}
    
    # Set task-specific evaluation parameters
    if task_type == 'segmentation':
        config['evaluation']['metrics'] = ['dice', 'iou']
        config['evaluation']['threshold'] = config['evaluation'].get('threshold', 0.5)
    else:  # classification
        config['evaluation']['metrics'] = ['accuracy', 'f1', 'precision', 'recall']
        config['evaluation']['per_class_metrics'] = config['evaluation'].get('per_class_metrics', True)
        config['evaluation']['confusion_matrix'] = config['evaluation'].get('confusion_matrix', True)
    
    # Ensure consistency between model and data settings for mask as channel
    if task_type == 'classification':
        use_mask = (
            config.get('data', {}).get('use_mask_as_channel', False) or 
            config.get('model', {}).get('use_mask_as_channel', False)
        )
        if use_mask:
            if 'data' not in config:
                config['data'] = {}
            if 'model' not in config:
                config['model'] = {}
            config['data']['use_mask_as_channel'] = True
            config['model']['use_mask_as_channel'] = True
    
    return config