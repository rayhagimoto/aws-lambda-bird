# AWS Lambda Bird Detection

A modular package for deploying bird anomaly detection models on AWS Lambda.

## Overview

This package contains implementations of bird anomaly detection models optimized for AWS Lambda deployment. It includes:

- **ConvDetector**: Full convolutional autoencoder-based anomaly detection with rolling window training
- **SimpleDetector**: Minimal anomaly detection using exponential moving average

## Branches

- `main`: **This branch** - Contains both detectors with all dependencies
- `conv-ae`: Specialized branch with ConvDetector only (includes PyTorch)
- `simple`: Specialized branch with SimpleDetector only (minimal dependencies)

## Quick Start

```python
from bird_detection import ConvDetector, SimpleDetector
import boto3

# Initialize S3 client
s3 = boto3.client('s3')

# Configuration
config = {
    'bucket_name': 'your-bucket',
    'state_folder': 'bird-detection-state',
    'incubation_period': 200,
    'enable_training': True
}

# Initialize detector (choose one)
detector = ConvDetector(config, s3)  # or SimpleDetector(config, s3)

# Predict anomaly
from PIL import Image
image = Image.open('bird.jpg')
is_anomaly = detector.predict(image)
```

## Lambda Function Usage

The lambda function supports both detectors via the `detector_type` parameter:

```json
{
    "image_url": "s3://your-bucket/image.jpg",
    "detector_type": "conv",  // or "simple"
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

## Detector Comparison

| Feature | ConvDetector | SimpleDetector |
|---------|-------------|----------------|
| **Dependencies** | PyTorch, TorchVision | NumPy, PIL only |
| **Memory Usage** | Higher (model weights) | Lower |
| **Speed** | Slower (neural network) | Faster |
| **Accuracy** | Higher (learned features) | Lower (statistical) |
| **Training** | Rolling window adaptation | Exponential moving average |
| **Burn-in Period** | Configurable (default: 200) | Minimum observations |

## Use Cases

- **ConvDetector**: Production environments with sufficient resources, high accuracy requirements
- **SimpleDetector**: Resource-constrained environments, rapid prototyping, edge devices

## License

MIT License 