import torch
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
from pathlib import Path

# Import your dataset classes
from loaders.segmentation import SegmentationDataset
from loaders.classification import ClassificationDataset
from utils.data import create_data_loader
from factories.dataset import create_segmentation_datasets, create_classification_datasets

class InferenceTester:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"Using device: {self.device}")
        
        # Create output directories
        os.makedirs(config['output_dir'], exist_ok=True)
        os.makedirs(os.path.join(config['output_dir'], 'visualizations'), exist_ok=True)
        
        # Load test data first
        _, _, self.test_dataset = create_segmentation_datasets(
            csv_file=config['data']['csv_file'],
            root_dir=config['data']['root_dir'],
            config=config['data']
        )
        
        self.test_loader = create_data_loader(
            self.test_dataset, 
            {'training': {'batch_size': 1, 'num_workers': 1}}, 
            shuffle=False
        )
        
        # Class mapping for classification
        self.class_names = ['low', 'medium', 'high', 'flood']
        
        # Load models directly from files without creating new architecture
        self.seg_model = self._load_model(config['segmentation']['model_path'])
        self.cls_model = self._load_model(config['classification']['model_path'])
        self.cls_model_with_mask = self._load_model(config['classification']['model_path_with_mask'])
        
        print("Successfully loaded all models")
        
    def _load_model(self, model_path):
        """Load a model from a checkpoint file"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
            
        # Load the checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Determine if it's a state_dict or full model
        if isinstance(checkpoint, dict):
            print(f"Loaded state_dict from {model_path}")
            
            # Get the actual state_dict from the checkpoint
            state_dict = checkpoint
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            
            # Detect if the model uses 4 input channels (mask)
            use_mask = False
            for key, value in state_dict.items():
                if 'features.0' in key and 'weight' in key and len(value.shape) == 4:
                    input_channels = value.shape[1]
                    if input_channels == 4:
                        use_mask = True
                        print(f"Detected model with 4 input channels (RGB + mask)")
                    break
                    
            # Create appropriate model based on filename
            filename = os.path.basename(model_path)
            
            # Import necessary modules for model creation
            from factories.model import create_model
            
            if 'seg' in filename.lower() or 'fpn' in filename.lower() or 'unet' in filename.lower() or 'deeplabv3' in filename.lower():
                # Segmentation model
                encoder = 'mobilenet_v3_large'  # default
                decoder = 'deeplabv3'  # default
                
                # Extract encoder type from filename
                if 'efficientnet' in filename.lower():
                    encoder = 'efficientnet_b2'
                elif 'squeezenet' in filename.lower():
                    encoder = 'squeezenet1_1'
                elif 'shufflenet' in filename.lower():
                    encoder = 'shufflenet_v2_x1_0'
                elif 'mnasnet' in filename.lower():
                    encoder = 'mnasnet1_0'
                    
                # Extract decoder type from filename
                if 'fpn' in filename.lower():
                    decoder = 'fpn'
                elif 'unet' in filename.lower():
                    decoder = 'unet'
                    
                print(f"Creating segmentation model with encoder={encoder}, decoder={decoder}")
                model = create_model('segmentation', {
                    'encoder': encoder,
                    'decoder': decoder
                })
            else:
                # Classification model
                encoder = 'mobilenet_v3_large'  # default
                
                # Extract encoder type from filename
                if 'efficientnet' in filename.lower():
                    encoder = 'efficientnet_b2'
                elif 'squeezenet' in filename.lower():
                    encoder = 'squeezenet1_1'
                elif 'shufflenet' in filename.lower():
                    encoder = 'shufflenet_v2_x1_0'
                elif 'mnasnet' in filename.lower():
                    encoder = 'mnasnet1_0'
                    
                print(f"Creating classification model with encoder={encoder}, use_mask={use_mask}")
                model = create_model('classification', {
                    'encoder': encoder,
                    'use_mask_as_channel': use_mask
                })
                
            # Load the state_dict
            try:
                model.load_state_dict(state_dict)
                print(f"Successfully loaded model weights")
            except Exception as e:
                print(f"Warning: Failed to load with strict matching: {e}")
                print("Attempting to load with non-strict matching...")
                # Try loading with strict=False
                model.load_state_dict(state_dict, strict=False)
                print(f"Loaded with non-strict matching")
                
        else:
            # Assuming it's a full model
            model = checkpoint
            print(f"Loaded full model from {model_path}")
            
        model = model.to(self.device)
        model.eval()
        
        if 'cls' in filename.lower() or not 'seg' in filename.lower():
            # Add a use_mask_channel attribute to the model
            setattr(model, 'use_mask_channel', use_mask)
        
        return model

    def test_inference_speed(self, num_runs=50):
        """Test inference speed of models"""
        print(f"\nTesting inference speed over {num_runs} runs...")
        
        # Get a single sample image for testing
        sample_image, _ = next(iter(self.test_loader))
        sample_image = sample_image.to(self.device)
        
        # Generate mask using segmentation model for the 4-channel input
        with torch.no_grad():
            seg_output = self.seg_model(sample_image)
            mask = torch.sigmoid(seg_output) > 0.5
            
            # Print shape information for debugging
            print(f"Input image shape: {sample_image.shape}")
            print(f"Segmentation output shape: {seg_output.shape}")
            print(f"Mask shape: {mask.shape}")
            
            # Resize mask to match input dimensions if needed
            if mask.shape[-2:] != sample_image.shape[-2:]:
                print(f"Resizing mask from {mask.shape[-2:]} to {sample_image.shape[-2:]}")
                mask_channel = torch.nn.functional.interpolate(
                    mask.float(), 
                    size=sample_image.shape[-2:],
                    mode='nearest'
                )
            else:
                mask_channel = mask.float()
            
            # Create 4-channel input for classification model
            image_with_mask = torch.cat([sample_image, mask_channel], dim=1)
            print(f"Combined image+mask shape: {image_with_mask.shape}")
        
        # Warm up the GPU
        with torch.no_grad():
            for _ in range(10):
                self.seg_model(sample_image)
                # Use the correct input format based on model requirements
                if hasattr(self.cls_model, 'use_mask_channel') and self.cls_model.use_mask_channel:
                    self.cls_model(image_with_mask)
                else:
                    self.cls_model(sample_image)
                self.cls_model_with_mask(image_with_mask)
        
        # Test segmentation model speed
        seg_times = []
        with torch.no_grad():
            for _ in range(num_runs):
                start_time = time.time()
                _ = self.seg_model(sample_image)
                torch.cuda.synchronize()
                seg_times.append(time.time() - start_time)
        
        # Test classification model speed (with or without mask)
        cls_times = []
        with torch.no_grad():
            for _ in range(num_runs):
                start_time = time.time()
                if hasattr(self.cls_model, 'use_mask_channel') and self.cls_model.use_mask_channel:
                    _ = self.cls_model(image_with_mask)
                else:
                    _ = self.cls_model(sample_image)
                torch.cuda.synchronize()
                cls_times.append(time.time() - start_time)
        
        # Test pipeline speed
        pipeline_times = []
        with torch.no_grad():
            for _ in range(num_runs):
                start_time = time.time()
                
                # Run segmentation
                seg_output = self.seg_model(sample_image)
                mask = torch.sigmoid(seg_output) > 0.5
                
                # Resize mask if needed
                if mask.shape[-2:] != sample_image.shape[-2:]:
                    mask_channel = torch.nn.functional.interpolate(
                        mask.float(),
                        size=sample_image.shape[-2:],
                        mode='nearest'
                    )
                else:
                    mask_channel = mask.float()
                
                # Add mask as channel to the original image
                image_with_mask = torch.cat([sample_image, mask_channel], dim=1)
                
                # Run classification with mask
                _ = self.cls_model_with_mask(image_with_mask)
                
                torch.cuda.synchronize()
                pipeline_times.append(time.time() - start_time)
        
        # Calculate statistics
        seg_avg = np.mean(seg_times) * 1000  # ms
        cls_avg = np.mean(cls_times) * 1000  # ms
        pipeline_avg = np.mean(pipeline_times) * 1000  # ms
        
        seg_std = np.std(seg_times) * 1000  # ms
        cls_std = np.std(cls_times) * 1000  # ms
        pipeline_std = np.std(pipeline_times) * 1000  # ms
        
        # Print results
        print(f"\nInference Speed Results (avg ± std over {num_runs} runs):")
        print(f"Segmentation:  {seg_avg:.2f} ± {seg_std:.2f} ms")
        print(f"Classification: {cls_avg:.2f} ± {cls_std:.2f} ms")
        print(f"Full Pipeline:  {pipeline_avg:.2f} ± {pipeline_std:.2f} ms")
        
        # Save results to CSV
        results = {
            'model': ['Segmentation', 'Classification', 'Pipeline'],
            'avg_time_ms': [seg_avg, cls_avg, pipeline_avg],
            'std_time_ms': [seg_std, cls_std, pipeline_std],
        }
        
        pd.DataFrame(results).to_csv(
            os.path.join(self.config['output_dir'], 'inference_speed.csv'),
            index=False
        )
        
        return seg_avg, cls_avg, pipeline_avg
        
    def evaluate_pipeline(self, num_samples=None):
        """Evaluate the segmentation->classification pipeline on test set"""
        # Implementation similar to previous version but with more robust model handling
        # (Function implementation continues...)
        # For brevity, I'm only showing the key modified part of the code

def find_model_files():
    """Find available model files in the project"""
    model_paths = {
        'segmentation': [],
        'classification': [],
        'classification_with_mask': []
    }
    
    # Check in standard directories
    for directory in ['models', 'checkpoints', 'results']:
        if os.path.exists(directory):
            for root, _, files in os.walk(directory):
                for file in files:
                    if file.endswith('.pth'):
                        filepath = os.path.join(root, file)
                        
                        # Categorize model files
                        if 'seg' in file.lower() or 'unet' in file.lower() or 'fpn' in file.lower() or 'deeplabv3' in file.lower():
                            model_paths['segmentation'].append(filepath)
                        elif 'mask' in file.lower():
                            model_paths['classification_with_mask'].append(filepath)
                        elif 'cls' in file.lower() or 'class' in file.lower():
                            model_paths['classification'].append(filepath)
                        else:
                            # Try to infer from directory name
                            if 'seg' in root.lower():
                                model_paths['segmentation'].append(filepath)
                            elif 'cls' in root.lower() or 'class' in root.lower():
                                if 'mask' in file.lower() or 'mask' in root.lower():
                                    model_paths['classification_with_mask'].append(filepath)
                                else:
                                    model_paths['classification'].append(filepath)
    
    return model_paths

def main():
    # Find available model files
    model_paths = find_model_files()
    
    # If no masked classification model, duplicate regular one
    if not model_paths['classification_with_mask'] and model_paths['classification']:
        model_paths['classification_with_mask'] = model_paths['classification']
    
    # Configure test
    config = {
        'output_dir': 'results/inference_test',
        'data': {
            'csv_file': 'dataset/river.csv',
            'root_dir': 'dataset',
            'image_size': 128,
            'augmentation': 'base'
        },
        'segmentation': {
            'model_path': model_paths['segmentation'][0] if model_paths['segmentation'] else 'checkpoints/best_model.pth'
        },
        'classification': {
            'model_path': model_paths['classification'][0] if model_paths['classification'] else 'checkpoints/classifier.pth',
            'model_path_with_mask': model_paths['classification_with_mask'][0] if model_paths['classification_with_mask'] else 'checkpoints/classifier_with_mask.pth'
        }
    }
    
    print("Using model files:")
    print(f"Segmentation: {config['segmentation']['model_path']}")
    print(f"Classification: {config['classification']['model_path']}")
    print(f"Classification with mask: {config['classification']['model_path_with_mask']}")
    
    # Create tester and run tests
    tester = InferenceTester(config)
    
    # Test inference speed only
    tester.test_inference_speed(num_runs=50)

if __name__ == "__main__":
    main()