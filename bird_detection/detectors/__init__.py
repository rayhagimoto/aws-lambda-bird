"""
Detector modules for bird anomaly detection.
"""

from .anomaly_detector import AnomalyDetector
from .convae_detector import ConvDetector
from .simple_detector import SimpleDetector

__all__ = ["AnomalyDetector", "ConvDetector", "SimpleDetector"] 