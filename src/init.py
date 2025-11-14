"""
Hospital Readmission Prediction System
======================================
A machine learning system for predicting 30-day hospital readmissions.

Author: [Your Name]
Date: November 2025
"""

__version__ = "1.0.0"
__author__ = "Your Name"

from .data_preprocessing import DataPreprocessor
from .model import ReadmissionPredictor
from .evaluation import ModelEvaluator

__all__ = ['DataPreprocessor', 'ReadmissionPredictor', 'ModelEvaluator']