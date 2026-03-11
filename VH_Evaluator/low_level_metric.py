#!/usr/bin/env python3
"""
Multi-folder Image Quality Metrics Analyzer
Generates separate JSON result files for different folders.
"""

import cv2
import numpy as np
import json
import os
from pathlib import Path
import sys
import argparse


def calculate_raw_metrics(gray_img):
    """Calculate raw values of four core metrics (without normalization)."""
    try:
        # 1. Sharpness - Laplacian variance
        laplacian = cv2.Laplacian(gray_img, cv2.CV_64F)
        sharpness_raw = laplacian.var()
        
        # 2. High-frequency energy - Mean absolute value after high-pass filtering
        kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=np.float32)
        high_freq = cv2.filter2D(gray_img.astype(np.float32), -1, kernel)
        high_freq_energy = np.mean(np.abs(high_freq))
        
        # 3. Edge artifact - Contrast in edge regions
        edges = cv2.Canny(gray_img, 50, 150)
        kernel_dilate = np.ones((3, 3), np.uint8)
        dilated_edges = cv2.dilate(edges, kernel_dilate, iterations=2)
        
        edge_mask = dilated_edges > 0
        if np.sum(edge_mask) > 0:
            edge_artifact = np.std(gray_img[edge_mask])
        else:
            edge_artifact = 0
        
        # 4. Noise level - Noise intensity in smooth regions
        blurred = cv2.GaussianBlur(gray_img, (5, 5), 0)
        noise_map = np.abs(gray_img.astype(np.float32) - blurred.astype(np.float32))
        
        kernel_size = 15
        kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
        
        local_mean = cv2.filter2D(gray_img.astype(np.float32), -1, kernel)
        local_sq_mean = cv2.filter2D((gray_img.astype(np.float32))**2, -1, kernel)
        local_variance = local_sq_mean - local_mean**2
        local_std = np.sqrt(np.maximum(local_variance, 0))
        
        smooth_threshold = np.percentile(local_std, 30)
        smooth_regions = local_std < smooth_threshold
        
        if np.sum(smooth_regions) > 0:
            noise_raw = np.mean(noise_map[smooth_regions])
        else:
            noise_raw = np.mean(noise_map)
        
        return sharpness_raw, high_freq_energy, edge_artifact, noise_raw
    except:
        return 0.0, 0.0, 0.0, 0.0


def analyze_directory(input_dir, show_details=False):
    """分析目录中的图像"""
    folder_name = Path(input_dir).name
    print("=" * 70)
    print(f"图像质量指标分析器 - {folder_name} 数据集")
    print("=" * 70)
    print(f"输入目录: {input_dir}")
    
    if not os.path.exists(input_dir):
        print(f"❌ Error: Directory does not exist - {input_dir}")
        return None, folder_name
    
    # Find image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(Path(input_dir).glob(f'*{ext}'))
        image_files.extend(Path(input_dir).glob(f'*{ext.upper()}'))
    
    if not image_files:
        print("❌ No supported image files found")
        return None, folder_name
    
    print(f"📊 Found {len(image_files)} images")
    print("-" * 70)
    
    # Analyze images
    sharpness_values = []
    high_freq_values = []
    edge_artifact_values = []
    noise_values = []
    successful_count = 0
    
    for i, image_file in enumerate(image_files, 1):
        try:
            if show_details or i <= 10:
                print(f"[{i:3d}/{len(image_files)}] {image_file.name}", end=" ... ")
            
            img = cv2.imread(str(image_file))
            if img is None:
                if show_details or i <= 10:
                    print("❌ Failed to read")
                continue
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            sharpness, high_freq, edge_artifact, noise = calculate_raw_metrics(gray)
            
            sharpness_values.append(sharpness)
            high_freq_values.append(high_freq)
            edge_artifact_values.append(edge_artifact)
            noise_values.append(noise)
            successful_count += 1
            
            if show_details or i <= 10:
                print(f"✓ Sharpness:{sharpness:.1f} HighFreq:{high_freq:.1f} Edge:{edge_artifact:.1f} Noise:{noise:.2f}")
            elif i % 50 == 0:
                print(f"[{i:3d}/{len(image_files)}] Processed...")
        
        except Exception as e:
            if show_details or i <= 10:
                print(f"❌ Error: {str(e)}")
    
    print("-" * 70)
    
    if successful_count == 0:
        print("❌ No images were successfully analyzed")
        return None, folder_name
    
    # Calculate averages and statistics
    avg_sharpness = sum(sharpness_values) / len(sharpness_values)
    avg_high_freq = sum(high_freq_values) / len(high_freq_values)
    avg_edge_artifact = sum(edge_artifact_values) / len(edge_artifact_values)
    avg_noise = sum(noise_values) / len(noise_values)
    
    # Output results
    print(f"📈 {folder_name} Dataset Analysis Results")
    print("-" * 40)
    print(f"Successfully analyzed images: {successful_count}")
    print()
    print("Average values of four core metrics:")
    print(f"  Average Sharpness: {avg_sharpness:.2f}")
    print(f"  Average High-Frequency Energy: {avg_high_freq:.2f}")
    print(f"  Average Edge Artifact: {avg_edge_artifact:.2f}")
    print(f"  Average Noise Level: {avg_noise:.3f}")
    
    # Display distribution statistics
    print(f"\nSharpness Distribution:")
    print(f"  Min: {min(sharpness_values):.2f}")
    print(f"  Max: {max(sharpness_values):.2f}")
    print(f"  Median: {np.median(sharpness_values):.2f}")
    print(f"  Std: {np.std(sharpness_values):.2f}")
    
    print(f"\nHigh-Frequency Energy Distribution:")
    print(f"  Min: {min(high_freq_values):.2f}")
    print(f"  Max: {max(high_freq_values):.2f}")
    print(f"  Median: {np.median(high_freq_values):.2f}")
    print(f"  Std: {np.std(high_freq_values):.2f}")
    
    print(f"\nEdge Artifact Distribution:")
    print(f"  Min: {min(edge_artifact_values):.2f}")
    print(f"  Max: {max(edge_artifact_values):.2f}")
    print(f"  Median: {np.median(edge_artifact_values):.2f}")
    print(f"  Std: {np.std(edge_artifact_values):.2f}")
    
    print(f"\nNoise Level Distribution:")
    print(f"  Min: {min(noise_values):.3f}")
    print(f"  Max: {max(noise_values):.3f}")
    print(f"  Median: {np.median(noise_values):.3f}")
    print(f"  Std: {np.std(noise_values):.3f}")
    
    print("=" * 70)
    
    return {
        'dataset_name': folder_name,
        'analysis_info': {
            'total_images': successful_count,
            'input_directory': input_dir,
            'note': 'All metrics are raw values without normalization'
        },
        'core_metrics_averages': {
            'sharpness': float(round(avg_sharpness, 2)),
            'high_frequency_energy': float(round(avg_high_freq, 2)),
            'edge_artifact': float(round(avg_edge_artifact, 2)),
            'noise_level': float(round(avg_noise, 3))
        },
        'sharpness_distribution': {
            'min': float(round(min(sharpness_values), 2)),
            'max': float(round(max(sharpness_values), 2)),
            'median': float(round(np.median(sharpness_values), 2)),
            'std': float(round(np.std(sharpness_values), 2))
        },
        'high_frequency_energy_distribution': {
            'min': float(round(min(high_freq_values), 2)),
            'max': float(round(max(high_freq_values), 2)),
            'median': float(round(np.median(high_freq_values), 2)),
            'std': float(round(np.std(high_freq_values), 2))
        },
        'edge_artifact_distribution': {
            'min': float(round(min(edge_artifact_values), 2)),
            'max': float(round(max(edge_artifact_values), 2)),
            'median': float(round(np.median(edge_artifact_values), 2)),
            'std': float(round(np.std(edge_artifact_values), 2))
        },
        'noise_distribution': {
            'min': float(round(min(noise_values), 3)),
            'max': float(round(max(noise_values), 3)),
            'median': float(round(np.median(noise_values), 3)),
            'std': float(round(np.std(noise_values), 3))
        }
    }, folder_name


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Multi-folder Image Quality Metrics Analyzer')
    parser.add_argument('input_dir', help='Input image folder path')
    parser.add_argument('output_dir', help='Output results save path')
    parser.add_argument('--verbose', action='store_true', help='Show detailed analysis information')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    result, folder_name = analyze_directory(args.input_dir, args.verbose)
    
    if result:
        # Generate different JSON files based on folder name
        output_file = os.path.join(args.output_dir, f'metrics_{folder_name.lower()}.json')
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        print(f"💾 Results saved to: {output_file}")
        print("✅ Analysis complete!")
        
        # Output brief summary
        print(f"\n📊 {folder_name} Dataset Core Metrics:")
        print(f"   Sharpness: {result['core_metrics_averages']['sharpness']}")
        print(f"   High-Frequency Energy: {result['core_metrics_averages']['high_frequency_energy']}")
        print(f"   Edge Artifact: {result['core_metrics_averages']['edge_artifact']}")
        print(f"   Noise Level: {result['core_metrics_averages']['noise_level']}")
    else:
        print("❌ Analysis failed")


if __name__ == "__main__":
    main()

