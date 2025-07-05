"""
Detector modules for bird anomaly detection.
"""

from .anomaly_detector import AnomalyDetector
from .convae_detector import ConvDetector

__all__ = ["AnomalyDetector", "ConvDetector"] 