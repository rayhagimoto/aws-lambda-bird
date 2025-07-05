"""
AWS Lambda function handler for bird anomaly detection (ConvDetector).
"""

import json
import boto3
import os
import requests
from PIL import Image
from io import BytesIO

from bird_detection import ConvDetector


def send_telegram_notification(image_url, detector_type="ConvDetector"):
    """Send Telegram notification when anomaly is detected."""
    try:
        telegram_bot_token = os.environ.get('TELEGRAM_BOT_TOKEN')
        telegram_chat_id = os.environ.get('TELEGRAM_CHAT_ID')
        
        if not telegram_bot_token or not telegram_chat_id:
            print("Warning: Telegram credentials not configured")
            return False
        
        text = f"🚨 Anomalous image detected!\n\nImage: {image_url}\nDetector: {detector_type}"
        
        response = requests.post(
            f"https://api.telegram.org/bot{telegram_bot_token}/sendMessage",
            data={
                "chat_id": telegram_chat_id, 
                "text": text,
                "parse_mode": "HTML"
            },
            timeout=10
        )
        
        if response.status_code == 200:
            print(f"Telegram notification sent successfully")
            return True
        else:
            print(f"Failed to send Telegram notification: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        print(f"Error sending Telegram notification: {e}")
        return False


def lambda_handler(event, context):
    """
    AWS Lambda handler for bird anomaly detection using ConvDetector.
    
    Expected event format:
    {
        "image_url": "s3://bucket-name/path/to/image.jpg",
        "config": {
            "bucket_name": "your-bucket",
            "state_folder": "bird-detection-state",
            "enable_training": true
        }
    }
    """
    
    try:
        # Parse event
        image_url = event.get('image_url')
        config = event.get('config', {})
        
        if not image_url:
            return {
                'statusCode': 400,
                'body': json.dumps({'error': 'image_url is required'})
            }
        
        # Initialize S3 client
        s3 = boto3.client('s3')
        
        # Load image from S3
        bucket, key = parse_s3_url(image_url)
        image = load_image_from_s3(s3, bucket, key)
        
        # Initialize ConvDetector
        detector = ConvDetector(config, s3)
        
        # Predict anomaly
        is_anomaly = detector.predict(image)
        
        # Send Telegram notification if anomaly detected
        notification_sent = False
        if is_anomaly:
            notification_sent = send_telegram_notification(image_url, "ConvDetector")
        
        # Get additional info if available
        result = {
            'anomaly_detected': is_anomaly,
            'detector_type': 'conv',
            'image_url': image_url,
            'notification_sent': notification_sent
        }
        
        # Add state information for ConvDetector
        if hasattr(detector, 'get_current_state'):
            state = detector.get_current_state()
            result['state'] = state
        
        return {
            'statusCode': 200,
            'body': json.dumps(result)
        }
        
    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': str(e),
                'type': type(e).__name__
            })
        }


def parse_s3_url(s3_url):
    """Parse S3 URL to get bucket and key."""
    if not s3_url.startswith('s3://'):
        raise ValueError("URL must start with s3://")
    
    # Remove s3:// prefix
    path = s3_url[5:]
    
    # Split on first slash
    parts = path.split('/', 1)
    if len(parts) != 2:
        raise ValueError("Invalid S3 URL format")
    
    bucket, key = parts
    return bucket, key


def load_image_from_s3(s3_client, bucket, key):
    """Load image from S3 and return PIL Image."""
    try:
        response = s3_client.get_object(Bucket=bucket, Key=key)
        image_data = response['Body'].read()
        image = Image.open(BytesIO(image_data)).convert('RGB')
        return image
    except Exception as e:
        raise Exception(f"Failed to load image from S3: {e}")


# For local testing
if __name__ == "__main__":
    # Test event
    test_event = {
        "image_url": "s3://your-bucket/test-image.jpg",
        "config": {
            "bucket_name": "your-bucket",
            "state_folder": "bird-detection-state",
            "enable_training": True
        }
    }
    
    result = lambda_handler(test_event, None)
    print(json.dumps(result, indent=2)) 