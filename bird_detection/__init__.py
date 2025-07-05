"""
AWS Lambda Bird Detection Package

A minimal package for deploying bird anomaly detection models on AWS Lambda.
"""

from .detectors.convae_detector import ConvDetector
from .detectors.simple_detector import SimpleDetector

__version__ = "1.0.0"
__all__ = ["ConvDetector", "SimpleDetector"] 