# AWS Lambda Bird Detection

A minimal package for deploying bird anomaly detection models on AWS Lambda.

## Overview

This package contains minimal implementations of bird anomaly detection models optimized for AWS Lambda deployment. It includes:

- **ConvDetector**: Full convolutional autoencoder-based anomaly detection with rolling window training
- **SimpleDetector**: Minimal anomaly detection for basic use cases

## Branches

- `main`: Default branch with basic structure
- `conv-ae`: Full ConvDetector implementation with rolling window training
- `simple`: Minimal SimpleDetector implementation

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

# Initialize detector
detector = ConvDetector(config, s3)

# Predict anomaly
from PIL import Image
image = Image.open('bird.jpg')
is_anomaly = detector.predict(image)
```

## Installation

```bash
pip install -r requirements.txt
```

## Deployment

See `DEPLOYMENT_GUIDE.md` for detailed AWS Lambda deployment instructions.

## License

MIT License 