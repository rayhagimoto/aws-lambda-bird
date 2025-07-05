# AWS Lambda Bird Detection (ConvDetector)

A specialized package for deploying ConvDetector on AWS Lambda.

## Overview

This package contains a specialized implementation of ConvDetector for bird anomaly detection, optimized for AWS Lambda deployment. It includes:

- **ConvDetector**: Full convolutional autoencoder-based anomaly detection with rolling window training

## Branches

- `main`: Contains both detectors with all dependencies
- `conv-ae`: **This branch** - Specialized ConvDetector only (includes PyTorch)
- `simple`: Specialized SimpleDetector only (minimal dependencies)

## Quick Start

```python
from bird_detection import ConvDetector
import boto3

# Initialize S3 client
s3 = boto3.client('s3')

# Configuration
config = {
    'bucket_name': 'your-bucket',
    'state_folder': 'bird-detection-state',
    'incubation_period': 200,
    'enable_training': True,
    'model': {
        'image_size': 64,
        'latent_dim': 2,
        'enc_channels': [16, 32, 64, 8],
        'dec_channels': [64, 32, 16, 3]
    }
}

# Initialize detector
detector = ConvDetector(config, s3)

# Predict anomaly
from PIL import Image
image = Image.open('bird.jpg')
is_anomaly = detector.predict(image)
```

## Lambda Function Usage

The lambda function is optimized for ConvDetector:

```json
{
    "image_url": "s3://your-bucket/image.jpg",
    "config": {
        "bucket_name": "your-bucket",
        "state_folder": "bird-detection-state",
        "enable_training": true
    }
}
```

## Installation

```bash
pip install -r requirements.txt
```

## Deployment

See `DEPLOYMENT_GUIDE.md` for detailed AWS Lambda deployment instructions.

## ConvDetector Features

- **Rolling Window Training**: Continuously adapts to new data
- **Burn-in Period**: Configurable incubation period (default: 200 images)
- **Dynamic Thresholding**: Uses Otsu method and percentile-based thresholds
- **State Persistence**: Saves model weights and scores to S3
- **Memory Optimized**: Configurable window sizes for Lambda constraints

## Configuration Parameters

- `incubation_period`: Images before anomaly detection starts (default: 200)
- `enable_training`: Enable/disable model training (default: true)
- `max_window_size`: Maximum images in rolling window (default: 200)
- `min_batch_size`: Minimum batch size for training (default: 32)
- `steps_per_image`: Training steps per image (default: 2)

## Model Architecture

- **Encoder**: 4 convolutional layers with max pooling
- **Latent Space**: Configurable dimension (default: 2)
- **Decoder**: 4 transposed convolutional layers
- **Activation**: ReLU for hidden layers, Sigmoid for output

## Performance Considerations

- **Memory**: ~50MB for model weights + window storage
- **Runtime**: ~2-5 seconds per image (including training)
- **Storage**: Model weights saved every 50 images

## License

MIT License 