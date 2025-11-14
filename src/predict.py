"""
Model Training Module for Hospital Readmission Prediction
==========================================================
This module contains the machine learning model implementation,
training procedures, and hyperparameter tuning for predicting
30-day hospital readmissions.

Author: [Your Name]
Date: November 2025
Course: AI for Software Engineering
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
import joblib
from typing import Tuple, Dict
import warnings
warnings.filterwarnings('ignore')


class ReadmissionPredictor:
    """
    A class to handle model training, validation, and prediction for 
    hospital readmission risk assessment.
    
    Attributes:
        model: The trained machine learning model
        best_params: Best hyperparameters found during tuning
        feature_importance: Dictionary of feature importance scores
    """
    
    def __init__(self, model_type: str = 'logistic_regression'):
        """
        Initialize the predictor with specified model type.
        
        Args:
            model_type: Type of model to use ('logistic_regression', 'random_forest', etc.)
        """
        self.model_type = model_type
        self.model = None
        self.best_params = None
        self.feature_importance = None
        self.feature_names = None
        
    def create_model(self, **kwargs) -> object:
        """
        Create and return a model instance based on model_type.
        
        For this assignment, we use Logistic Regression with L2 regularization
        because:
        1. Highly interpretable for clinical decision support
        2. Provides probability estimates for risk stratification
        3. Fast training and inference suitable for real-time deployment
        4. Well-studied in healthcare literature
        5. Regularization prevents overfitting
        
        Args:
            **kwargs: Additional model parameters
            
        Returns:
            Initialized model object
        """
        if self.model_type == 'logistic_regression':
            model = LogisticRegression(
                penalty='l2',  # L2 regularization to prevent overfitting
                solver='lbfgs',  # Optimization algorithm
                max_iter=1000,  # Maximum iterations for convergence
                random_state=42,  # For reproducibility
                class_weight='balanced',  # Handle class imbalance
                **kwargs
            )
            print("Created Logistic Regression model with L2 regularization")
            
        else:
            raise ValueError(f"Model type {self.model_type} not implemented")
            
        return model
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series,
              X_val: pd.DataFrame = None, y_val: pd.Series = None) -> None:
        """
        Train the model on training data with optional validation monitoring.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
        """
        print("=== Starting Model Training ===")
        print(f"Training samples: {len(X_train)}")
        print(f"Features: {X_train.shape[1]}")
        print(f"Positive class ratio: {y_train.mean():.2%}")
        
        # Store feature names for later interpretation
        self.feature_names = X_train.columns.tolist()
        
        # Create model instance
        self.model = self.create_model()
        
        # Train the model
        self.model.fit(X_train, y_train)
        print("✓ Model training completed")
        
        # Evaluate on training data
        train_score = self.model.score(X_train, y_train)
        print(f"Training accuracy: {train_score:.4f}")
        
        # Evaluate on validation data if provided
        if X_val is not None and y_val is not None:
            val_score = self.model.score(X_val, y_val)
            print(f"Validation accuracy: {val_score:.4f}")
            
            # Check for overfitting
            if train_score - val_score > 0.1:
                print("⚠ Warning: Potential overfitting detected")
                print(f"   Training-Validation gap: {train_score - val_score:.4f}")
        
        # Extract feature importance (coefficients for logistic regression)
        if hasattr(self.model, 'coef_'):
            self._extract_feature_importance()
    
    def _extract_feature_importance(self) -> None:
        """
        Extract and store feature importance from the trained model.
        For logistic regression, this is the absolute value of coefficients.
        """
        if self.model is None:
            print("Error: Model not trained yet")
            return
        
        # Get coefficients
        coefficients = self.model.coef_[0]
        
        # Create dictionary of feature importance
        self.feature_importance = {
            feature: abs(coef) 
            for feature, coef in zip(self.feature_names, coefficients)
        }
        
        # Sort by importance
        self.feature_importance = dict(
            sorted(self.feature_importance.items(), 
                   key=lambda x: x[1], 
                   reverse=True)
        )
        
        print("\n=== Top 10 Most Important Features ===")
        for i, (feature, importance) in enumerate(list(self.feature_importance.items())[:10], 1):
            print(f"{i:2d}. {feature:30s}: {importance:.4f}")
    
    def hyperparameter_tuning(self, X_train: pd.DataFrame, y_train: pd.Series,
                             cv_folds: int = 5) -> Dict:
        """
        Perform hyperparameter tuning using GridSearchCV with cross-validation.
        
        Hyperparameters tuned:
        1. C (inverse regularization strength): Lower values = stronger regularization
           - Tests: [0.001, 0.01, 0.1, 1, 10, 100]
           - Why: Controls model complexity and overfitting
        
        2. penalty: Type of regularization
           - Tests: ['l2'] (Ridge regression)
           - Why: Shrinks coefficients to prevent overfitting
        
        Args:
            X_train: Training features
            y_train: Training labels
            cv_folds: Number of cross-validation folds
            
        Returns:
            Dictionary of best hyperparameters
        """
        print("=== Starting Hyperparameter Tuning ===")
        print(f"Using {cv_folds}-fold cross-validation")
        
        # Define parameter grid to search
        param_grid = {
            'C': [0.001, 0.01, 0.1, 1, 10, 100],  # Regularization strength
            'penalty': ['l2'],  # Regularization type
            'solver': ['lbfgs'],  # Optimization algorithm
        }
        
        print(f"Testing {np.prod([len(v) for v in param_grid.values()])} parameter combinations")
        
        # Create base model
        base_model = self.create_model()
        
        # Setup GridSearchCV
        # scoring='f1' because we care about both precision and recall
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            cv=cv_folds,  # Cross-validation folds
            scoring='f1',  # F1-score balances precision and recall
            n_jobs=-1,  # Use all available processors
            verbose=1
        )
        
        # Perform grid search
        grid_search.fit(X_train, y_train)
        
        # Store best parameters
        self.best_params = grid_search.best_params_
        self.model = grid_search.best_estimator_
        
        print("\n=== Hyperparameter Tuning Complete ===")
        print(f"Best parameters: {self.best_params}")
        print(f"Best cross-validation F1-score: {grid_search.best_score_:.4f}")
        
        # Show performance of different parameter combinations
        print("\n=== Top 5 Parameter Combinations ===")
        results_df = pd.DataFrame(grid_search.cv_results_)
        top_results = results_df.nsmallest(5, 'rank_test_score')[
            ['params', 'mean_test_score', 'std_test_score']
        ]
        print(top_results.to_string(index=False))
        
        return self.best_params
    
    def cross_validate(self, X: pd.DataFrame, y: pd.Series, 
                      cv_folds: int = 5) -> np.ndarray:
        """
        Perform k-fold cross-validation to assess model generalization.
        
        Cross-validation helps:
        1. Estimate model performance on unseen data
        2. Detect overfitting by comparing train/validation scores
        3. Ensure model works well across different data subsets
        
        Args:
            X: Features
            y: Labels
            cv_folds: Number of folds (default: 5)
            
        Returns:
            Array of scores for each fold
        """
        if self.model is None:
            print("Creating new model for cross-validation")
            self.model = self.create_model()
        
        print(f"\n=== Performing {cv_folds}-Fold Cross-Validation ===")
        
        # Perform cross-validation with multiple metrics
        scoring_metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        
        for metric in scoring_metrics:
            scores = cross_val_score(
                self.model, X, y, 
                cv=cv_folds, 
                scoring=metric,
                n_jobs=-1
            )
            
            print(f"{metric.upper():12s}: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
            print(f"              Scores: {[f'{s:.3f}' for s in scores]}")
        
        return scores
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X: Features for prediction
            
        Returns:
            Array of predicted class labels (0 or 1)
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        predictions = self.model.predict(X)
        return predictions
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class probabilities for risk stratification.
        
        This is crucial for healthcare applications because:
        1. Clinicians can set custom thresholds based on risk tolerance
        2. Enables risk stratification (low/medium/high risk categories)
        3. Provides interpretable risk scores for patients
        
        Args:
            X: Features for prediction
            
        Returns:
            Array of shape (n_samples, 2) with probabilities for each class
            Column 0: probability of no readmission
            Column 1: probability of readmission
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        probabilities = self.model.predict_proba(X)
        return probabilities
    
    def save_model(self, filepath: str = 'models/readmission_model.pkl') -> None:
        """
        Save trained model to disk for deployment.
        
        Args:
            filepath: Path where model should be saved
        """
        if self.model is None:
            raise ValueError("No model to save. Train model first.")
        
        # Create directory if it doesn't exist
        import os
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save model and metadata
        model_data = {
            'model': self.model,
            'feature_names': self.feature_names,
            'feature_importance': self.feature_importance,
            'best_params': self.best_params,
            'model_type': self.model_type
        }
        
        joblib.dump(model_data, filepath)
        print(f"✓ Model saved to {filepath}")
    
    def load_model(self, filepath: str = 'models/readmission_model.pkl') -> None:
        """
        Load trained model from disk.
        
        Args:
            filepath: Path to saved model
        """
        try:
            model_data = joblib.load(filepath)
            self.model = model_data['model']
            self.feature_names = model_data['feature_names']
            self.feature_importance = model_data['feature_importance']
            self.best_params = model_data.get('best_params')
            self.model_type = model_data['model_type']
            print(f"✓ Model loaded from {filepath}")
        except FileNotFoundError:
            print(f"Error: Model file {filepath} not found")


# Example usage demonstrating the complete training pipeline
if __name__ == "__main__":
    print("="*60)
    print("HOSPITAL READMISSION PREDICTION MODEL TRAINING")
    print("="*60)
    
    # Load preprocessed data (assuming preprocessing is already done)
    # In practice, you would load from saved files or preprocessing pipeline
    from sklearn.datasets import make_classification
    
    # Generate synthetic data for demonstration
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_classes=2,
        weights=[0.85, 0.15],  # Imbalanced like real readmission data
        random_state=42
    )
    
    # Convert to DataFrame
    feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    X = pd.DataFrame(X, columns=feature_names)
    y = pd.Series(y, name='readmitted_30days')
    
    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    # Initialize predictor
    predictor = ReadmissionPredictor(model_type='logistic_regression')
    
    # Option 1: Quick training
    print("\n" + "="*60)
    print("OPTION 1: Quick Training")
    print("="*60)
    predictor.train(X_train, y_train, X_val, y_val)
    
    # Option 2: Training with hyperparameter tuning (recommended)
    print("\n" + "="*60)
    print("OPTION 2: Training with Hyperparameter Tuning")
    print("="*60)
    best_params = predictor.hyperparameter_tuning(X_train, y_train, cv_folds=5)
    
    # Cross-validation
    print("\n" + "="*60)
    print("CROSS-VALIDATION RESULTS")
    print("="*60)
    cv_scores = predictor.cross_validate(X_train, y_train, cv_folds=5)
    
    # Make predictions
    print("\n" + "="*60)
    print("MAKING PREDICTIONS")
    print("="*60)
    y_pred = predictor.predict(X_test)
    y_pred_proba = predictor.predict_proba(X_test)
    
    print(f"Predictions shape: {y_pred.shape}")
    print(f"Probabilities shape: {y_pred_proba.shape}")
    print(f"Sample predictions: {y_pred[:5]}")
    print(f"Sample probabilities (readmission risk):")
    for i in range(5):
        print(f"  Patient {i+1}: {y_pred_proba[i, 1]:.2%}")
    
    # Save model
    print("\n" + "="*60)
    print("SAVING MODEL")
    print("="*60)
    predictor.save_model('models/readmission_model.pkl')
    
    print("\n" + "="*60)
    print("TRAINING PIPELINE COMPLETE")
    print("="*60)
    print("Next step: Evaluate model performance using evaluation.py")