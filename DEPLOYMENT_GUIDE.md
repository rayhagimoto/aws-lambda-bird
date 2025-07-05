# AWS Lambda Deployment Guide

## Overview

This guide explains how to deploy the bird detection models on AWS Lambda.

## Prerequisites

- AWS CLI configured
- Python 3.9+ runtime
- S3 bucket for model state storage
- IAM permissions for Lambda and S3

## Package Structure

```
aws-lambda-bird/
├── bird_detection/
│   ├── __init__.py
│   ├── autoencoder/
│   │   ├── __init__.py
│   │   ├── convolution.py
│   │   └── scores.py
│   ├── detectors/
│   │   ├── __init__.py
│   │   ├── anomaly_detector.py
│   │   ├── convae_detector.py
│   │   └── simple_detector.py
│   └── utils.py
├── lambda_function.py
├── requirements.txt
└── README.md
```

## Deployment Steps

### 1. Create Lambda Function

```bash
# Create deployment package
zip -r lambda_package.zip . -x "*.git*" "*.pyc" "__pycache__/*"

# Create Lambda function
aws lambda create-function \
  --function-name bird-detection \
  --runtime python3.9 \
  --role arn:aws:iam::YOUR_ACCOUNT:role/lambda-execution-role \
  --handler lambda_function.lambda_handler \
  --zip-file fileb://lambda_package.zip \
  --timeout 300 \
  --memory-size 1024
```

### 2. Configure Environment Variables

```bash
aws lambda update-function-configuration \
  --function-name bird-detection \
  --environment Variables='{
    "BUCKET_NAME": "your-bucket-name",
    "STATE_FOLDER": "bird-detection-state",
    "ENABLE_TRAINING": "true"
  }'
```

### 3. Set IAM Permissions

Ensure your Lambda execution role has:
- S3 read/write permissions for your bucket
- CloudWatch Logs permissions

### 4. Test the Function

```bash
# Test with sample event
aws lambda invoke \
  --function-name bird-detection \
  --payload '{"image_url": "s3://your-bucket/test-image.jpg"}' \
  response.json
```

## Configuration

### Environment Variables

- `BUCKET_NAME`: S3 bucket for state storage
- `STATE_FOLDER`: Folder within bucket for state files
- `ENABLE_TRAINING`: Enable/disable model training (true/false)
- `INCUBATION_PERIOD`: Number of images before anomaly detection starts

### S3 State Files

The detector maintains state in S3:
- `model_weights.pth`: Current model weights
- `scores.npy`: Historical anomaly scores
- `img_window.npy`: Rolling window of recent images

## Performance Optimization

- Use Lambda layers for large dependencies
- Optimize memory allocation (1024MB recommended)
- Set appropriate timeout (300 seconds)
- Use S3 for large file storage

## Monitoring

- CloudWatch Logs for debugging
- CloudWatch Metrics for performance monitoring
- S3 access logs for state file tracking

## Troubleshooting

### Common Issues

1. **Memory Errors**: Increase Lambda memory allocation
2. **Timeout Errors**: Increase timeout or optimize model inference
3. **S3 Permission Errors**: Check IAM role permissions
4. **Import Errors**: Ensure all dependencies are included in package

### Debug Mode

Enable debug logging by setting environment variable:
```
DEBUG=true
```

## Cost Optimization

- Use Lambda Provisioned Concurrency for consistent performance
- Optimize S3 storage class for state files
- Monitor CloudWatch metrics for usage patterns 