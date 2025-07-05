"""
AWS Lambda Bird Detection Package (ConvDetector Specialized)

A minimal package for deploying ConvDetector on AWS Lambda.
"""

from .detectors.convae_detector import ConvDetector

__version__ = "1.0.0"
__all__ = ["ConvDetector"] 