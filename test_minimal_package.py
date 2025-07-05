#!/usr/bin/env python3
"""
Test script for the minimal bird detection package.
"""

import sys
import os
import numpy as np
from PIL import Image

# Add the package to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bird_detection import ConvDetector, SimpleDetector


def create_test_image(size=(64, 64)):
    """Create a test image for testing."""
    # Create a simple test image
    img_array = np.random.randint(0, 255, (size[0], size[1], 3), dtype=np.uint8)
    return Image.fromarray(img_array)


def test_simple_detector():
    """Test the SimpleDetector."""
    print("Testing SimpleDetector...")
    
    # Mock S3 client (you would use real boto3 in production)
    class MockS3:
        def get_object(self, Bucket, Key):
            raise Exception("NoSuchKey")
        def put_object(self, Bucket, Key, Body):
            pass
        exceptions = type('exceptions', (), {'NoSuchKey': Exception})()
    
    s3 = MockS3()
    
    # Configuration
    config = {
        'bucket_name': 'test-bucket',
        'state_folder': 'test-state',
        'image_size': [64, 64],
        'percentile': 98,
        'alpha': 0.05,
        'min_observations': 25
    }
    
    # Initialize detector
    detector = SimpleDetector(config, s3)
    
    # Test with multiple images
    for i in range(30):
        img = create_test_image()
        result = detector.predict(img)
        print(f"Image {i+1}: Anomaly = {result}")
    
    print("SimpleDetector test completed!\n")


def test_conv_detector():
    """Test the ConvDetector."""
    print("Testing ConvDetector...")
    
    # Mock S3 client
    class MockS3:
        def get_object(self, Bucket, Key):
            raise Exception("NoSuchKey")
        def put_object(self, Bucket, Key, Body):
            pass
        exceptions = type('exceptions', (), {'NoSuchKey': Exception})()
    
    s3 = MockS3()
    
    # Configuration
    config = {
        'bucket_name': 'test-bucket',
        'state_folder': 'test-state',
        'incubation_period': 5,  # Short for testing
        'enable_training': False,  # Disable training for testing
        'model': {
            'image_size': 64,
            'latent_dim': 2,
            'enc_channels': [16, 32, 64, 8],
            'enc_kernel_sizes': [3, 3, 3, 3],
            'dec_channels': [64, 32, 16, 3],
            'dec_kernel_sizes': [3, 3, 3, 3],
            'pool_kernel': 2,
            'upsample_mode': 'nearest',
        }
    }
    
    # Initialize detector
    detector = ConvDetector(config, s3)
    
    # Test with multiple images
    for i in range(10):
        img = create_test_image()
        result = detector.predict(img)
        print(f"Image {i+1}: Anomaly = {result}")
        
        # Get state info
        if hasattr(detector, 'get_current_state'):
            state = detector.get_current_state()
            print(f"  State: {state}")
    
    print("ConvDetector test completed!\n")


def test_imports():
    """Test that all imports work correctly."""
    print("Testing imports...")
    
    try:
        from bird_detection import ConvDetector, SimpleDetector
        print("✓ Main imports successful")
        
        from bird_detection.autoencoder import ConvAutoencoder, localized_reconstruction_score
        print("✓ Autoencoder imports successful")
        
        from bird_detection.detectors import AnomalyDetector
        print("✓ Detector imports successful")
        
        print("All imports successful!\n")
        return True
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 50)
    print("Testing Minimal Bird Detection Package")
    print("=" * 50)
    
    # Test imports first
    if not test_imports():
        print("Import tests failed. Exiting.")
        return
    
    # Test detectors
    try:
        test_simple_detector()
    except Exception as e:
        print(f"SimpleDetector test failed: {e}")
    
    try:
        test_conv_detector()
    except Exception as e:
        print(f"ConvDetector test failed: {e}")
    
    print("=" * 50)
    print("All tests completed!")
    print("=" * 50)


if __name__ == "__main__":
    main() 