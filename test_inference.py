#!/usr/bin/env python3
# filepath: /home/zelx/ic/test_inference.py

import os
import torch
import argparse
import time
from PIL import Image
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser(description='Test model inference')
    parser.add_argument('--task', type=str, required=True, 
                        choices=['segmentation', 'classification', 'combined'],
                        help='Task type (segmentation, classification, combined)')
    parser.add_argument('--model', type=str, required=True, 
                        help='Path to model checkpoint')
    parser.add_argument('--image', type=str, required=True,
                        help='Path to test image or directory of images')
    parser.add_argument('--output_dir', type=str, default='inference_results',
                        help='Directory to save results')
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Import appropriate inferencer
    if args.task == 'segmentation':
        from inference.segmentation import SegmentationInference
        inferencer = SegmentationInference(args.model, device)
    elif args.task == 'classification':
        from inference.classification import ClassificationInference
        inferencer = ClassificationInference(args.model, device)
    else:  # combined
        from inference.combined import CombinedInference
        inferencer = CombinedInference(args.model, device)
    
    # Run on single image or directory
    if os.path.isdir(args.image):
        print(f"Processing all images in directory: {args.image}")
        image_files = [f for f in os.listdir(args.image) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        # Process and time each image
        total_time = 0
        for img_file in image_files:
            img_path = os.path.join(args.image, img_file)
            output_path = os.path.join(args.output_dir, f"{os.path.splitext(img_file)[0]}_result.png")
            
            start_time = time.time()
            prediction = inferencer.predict(img_path)
            inference_time = time.time() - start_time
            total_time += inference_time
            
            inferencer.visualize(img_path, prediction, output_path)
            print(f"Processed {img_file} in {inference_time:.4f}s")
        
        print(f"\nTotal processing time: {total_time:.2f}s")
        print(f"Average time per image: {total_time/len(image_files):.4f}s")
        print(f"Results saved to {args.output_dir}")
    else:
        # Process single image
        output_path = os.path.join(args.output_dir, "result.png")
        
        start_time = time.time()
        prediction = inferencer.predict(args.image)
        inference_time = time.time() - start_time
        
        inferencer.visualize(args.image, prediction, output_path)
        print(f"Processed image in {inference_time:.4f}s")
        print(f"Result saved to {output_path}")
        
        # Display more details about the prediction
        print("\nPrediction details:")
        if args.task == 'segmentation':
            mask_ratio = prediction.mean()
            print(f"Mask coverage: {mask_ratio:.2%}")
        elif args.task == 'classification':
            print(f"Class: {prediction['class_name']}")
            print(f"Confidence: {prediction['confidence']:.2%}")
        else:  # combined
            print(f"Class: {prediction['class_name']}")
            print(f"Confidence: {prediction['confidence']:.2%}")
            print(f"Mask coverage: {prediction['mask'].mean():.2%}")

if __name__ == '__main__':
    main()