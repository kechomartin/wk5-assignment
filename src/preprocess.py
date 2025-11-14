"""
Data Preprocessing Module for Hospital Readmission Prediction
==============================================================
This module contains functions for cleaning, transforming, and preparing
patient data for machine learning model training.

Author: [Your Name]
Date: November 2025
Course: AI for Software Engineering
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from typing import Tuple, List

class DataPreprocessor:
    """
    A class to handle all data preprocessing steps for readmission prediction.
    
    Attributes:
        scaler: StandardScaler for numerical features
        label_encoders: Dictionary of LabelEncoders for categorical features
        imputer_numeric: SimpleImputer for numerical missing values
        imputer_categorical: SimpleImputer for categorical missing values
    """
    
    def __init__(self):
        """Initialize preprocessing components."""
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.imputer_numeric = SimpleImputer(strategy='mean')
        self.imputer_categorical = SimpleImputer(strategy='most_frequent')
        
    def load_data(self, filepath: str) -> pd.DataFrame:
        """
        Load patient data from CSV file.
        
        Args:
            filepath: Path to the CSV file containing patient data
            
        Returns:
            DataFrame containing raw patient data
            
        Example:
            >>> preprocessor = DataPreprocessor()
            >>> data = preprocessor.load_data('data/patient_records.csv')
        """
        try:
            df = pd.read_csv(filepath)
            print(f"Successfully loaded {len(df)} patient records")
            print(f"Columns: {df.columns.tolist()}")
            return df
        except FileNotFoundError:
            print(f"Error: File {filepath} not found")
            return None
            
    def handle_missing_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in the dataset using appropriate imputation strategies.
        
        Strategy:
        - Numerical features: Impute with mean
        - Categorical features: Impute with mode (most frequent)
        - Drop rows with >30% missing values
        
        Args:
            df: Input DataFrame with potential missing values
            
        Returns:
            DataFrame with missing values handled
        """
        # Calculate missing percentage per row
        missing_pct = df.isnull().sum(axis=1) / len(df.columns)
        
        # Drop rows with excessive missing data (>30%)
        df_clean = df[missing_pct < 0.3].copy()
        print(f"Dropped {len(df) - len(df_clean)} rows with >30% missing values")
        
        # Separate numerical and categorical columns
        numeric_cols = df_clean.select_dtypes(include=['int64', 'float64']).columns
        categorical_cols = df_clean.select_dtypes(include=['object']).columns
        
        # Impute numerical features with mean
        if len(numeric_cols) > 0:
            df_clean[numeric_cols] = self.imputer_numeric.fit_transform(df_clean[numeric_cols])
            print(f"Imputed {len(numeric_cols)} numerical features")
        
        # Impute categorical features with mode
        if len(categorical_cols) > 0:
            df_clean[categorical_cols] = self.imputer_categorical.fit_transform(df_clean[categorical_cols])
            print(f"Imputed {len(categorical_cols)} categorical features")
            
        return df_clean
    
    def feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create new features from existing data to improve model performance.
        
        Engineered Features:
        1. Charlson Comorbidity Index: Weighted score of patient comorbidities
        2. Polypharmacy Flag: Binary indicator if patient takes >5 medications
        3. Days Since Last Admission: Time-based feature
        4. Age Groups: Categorize age into risk bands
        
        Args:
            df: DataFrame with raw features
            
        Returns:
            DataFrame with additional engineered features
        """
        df_engineered = df.copy()
        
        # Feature 1: Charlson Comorbidity Index (simplified version)
        # This weights different conditions by severity
        comorbidity_weights = {
            'diabetes': 1,
            'heart_disease': 1,
            'copd': 1,
            'cancer': 2,
            'kidney_disease': 2
        }
        
        df_engineered['charlson_score'] = 0
        for condition, weight in comorbidity_weights.items():
            if condition in df.columns:
                df_engineered['charlson_score'] += df[condition] * weight
        
        print(f"Created Charlson Comorbidity Index")
        
        # Feature 2: Polypharmacy flag (taking more than 5 medications)
        if 'medication_count' in df.columns:
            df_engineered['polypharmacy'] = (df['medication_count'] > 5).astype(int)
            print(f"Created polypharmacy feature")
        
        # Feature 3: Days since last admission (if available)
        if 'last_admission_date' in df.columns and 'current_admission_date' in df.columns:
            df_engineered['days_since_last_admission'] = (
                pd.to_datetime(df['current_admission_date']) - 
                pd.to_datetime(df['last_admission_date'])
            ).dt.days
            print(f"Created days_since_last_admission feature")
        
        # Feature 4: Age groups for risk stratification
        if 'age' in df.columns:
            df_engineered['age_group'] = pd.cut(
                df['age'], 
                bins=[0, 45, 65, 75, 100],
                labels=['young', 'middle', 'senior', 'elderly']
            )
            print(f"Created age_group categories")
        
        # Feature 5: Interaction feature: age × comorbidity score
        if 'age' in df.columns and 'charlson_score' in df_engineered.columns:
            df_engineered['age_comorbidity_interaction'] = (
                df['age'] * df_engineered['charlson_score']
            )
            print(f"Created age-comorbidity interaction feature")
            
        return df_engineered
    
    def encode_categorical_features(self, df: pd.DataFrame, 
                                   categorical_cols: List[str]) -> pd.DataFrame:
        """
        Encode categorical variables for machine learning models.
        
        Strategy:
        - Binary categories (2 unique values): Label encoding (0, 1)
        - Multi-class categories: One-hot encoding
        - Ordinal categories: Manual mapping with order preservation
        
        Args:
            df: DataFrame with categorical features
            categorical_cols: List of column names to encode
            
        Returns:
            DataFrame with encoded categorical features
        """
        df_encoded = df.copy()
        
        for col in categorical_cols:
            if col not in df.columns:
                continue
                
            unique_values = df[col].nunique()
            
            # Binary encoding for binary categories
            if unique_values == 2:
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
                print(f"Label encoded {col} (2 categories)")
                
            # One-hot encoding for multi-class categories with few categories
            elif unique_values <= 10:
                dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
                df_encoded = pd.concat([df_encoded, dummies], axis=1)
                df_encoded.drop(col, axis=1, inplace=True)
                print(f"One-hot encoded {col} ({unique_values} categories)")
                
            # For high-cardinality features, use frequency encoding
            else:
                freq_map = df[col].value_counts(normalize=True).to_dict()
                df_encoded[f'{col}_frequency'] = df[col].map(freq_map)
                df_encoded.drop(col, axis=1, inplace=True)
                print(f"Frequency encoded {col} ({unique_values} categories)")
                
        return df_encoded
    
    def normalize_features(self, df: pd.DataFrame, 
                          numeric_cols: List[str]) -> pd.DataFrame:
        """
        Normalize numerical features using standardization (z-score normalization).
        
        Formula: z = (x - μ) / σ
        Where μ is mean and σ is standard deviation
        
        This ensures all features have mean=0 and std=1, preventing features
        with larger scales from dominating the model.
        
        Args:
            df: DataFrame with numerical features
            numeric_cols: List of numerical column names to normalize
            
        Returns:
            DataFrame with normalized numerical features
        """
        df_normalized = df.copy()
        
        # Only normalize columns that exist in the dataframe
        cols_to_normalize = [col for col in numeric_cols if col in df.columns]
        
        if cols_to_normalize:
            # Fit and transform the scaler
            df_normalized[cols_to_normalize] = self.scaler.fit_transform(
                df[cols_to_normalize]
            )
            print(f"Normalized {len(cols_to_normalize)} numerical features")
            print(f"Features: {cols_to_normalize}")
        
        return df_normalized
    
    def detect_and_handle_outliers(self, df: pd.DataFrame, 
                                   numeric_cols: List[str],
                                   method: str = 'iqr') -> pd.DataFrame:
        """
        Detect and handle outliers using Interquartile Range (IQR) method.
        
        IQR Method:
        - Calculate Q1 (25th percentile) and Q3 (75th percentile)
        - IQR = Q3 - Q1
        - Lower bound = Q1 - 1.5 * IQR
        - Upper bound = Q3 + 1.5 * IQR
        - Cap values outside bounds to the bounds (winsorization)
        
        Args:
            df: DataFrame with numerical features
            numeric_cols: List of columns to check for outliers
            method: Method to use ('iqr' or 'zscore')
            
        Returns:
            DataFrame with outliers handled
        """
        df_clean = df.copy()
        
        for col in numeric_cols:
            if col not in df.columns:
                continue
                
            if method == 'iqr':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # Count outliers
                outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
                
                # Cap outliers (winsorization)
                df_clean[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
                
                if outliers > 0:
                    print(f"Handled {outliers} outliers in {col}")
                    
        return df_clean
    
    def prepare_train_test_split(self, df: pd.DataFrame, 
                                target_col: str,
                                test_size: float = 0.15,
                                val_size: float = 0.15,
                                random_state: int = 42) -> Tuple[pd.DataFrame, ...]:
        """
        Split data into training, validation, and test sets with stratification.
        
        Split Strategy:
        - Training: 70% - for model learning
        - Validation: 15% - for hyperparameter tuning
        - Test: 15% - for final evaluation
        - Stratified to maintain class proportions
        
        Args:
            df: Complete preprocessed DataFrame
            target_col: Name of the target variable column
            test_size: Proportion for test set (0.15 = 15%)
            val_size: Proportion for validation set (0.15 = 15%)
            random_state: Random seed for reproducibility
            
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        from sklearn.model_selection import train_test_split
        
        # Separate features and target
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=random_state,
            stratify=y  # Maintain class proportions
        )
        
        # Second split: separate validation from training
        val_size_adjusted = val_size / (1 - test_size)  # Adjust for already split test
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_size_adjusted,
            random_state=random_state,
            stratify=y_temp
        )
        
        print(f"Data split completed:")
        print(f"  Training set: {len(X_train)} samples ({len(X_train)/len(df)*100:.1f}%)")
        print(f"  Validation set: {len(X_val)} samples ({len(X_val)/len(df)*100:.1f}%)")
        print(f"  Test set: {len(X_test)} samples ({len(X_test)/len(df)*100:.1f}%)")
        print(f"  Positive class in train: {y_train.mean():.2%}")
        print(f"  Positive class in val: {y_val.mean():.2%}")
        print(f"  Positive class in test: {y_test.mean():.2%}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test


# Example usage demonstrating the complete preprocessing pipeline
if __name__ == "__main__":
    # Initialize preprocessor
    preprocessor = DataPreprocessor()
    
    # Load data
    print("=== Step 1: Loading Data ===")
    df = preprocessor.load_data('data/patient_records.csv')
    
    # Handle missing data
    print("\n=== Step 2: Handling Missing Data ===")
    df_clean = preprocessor.handle_missing_data(df)
    
    # Feature engineering
    print("\n=== Step 3: Feature Engineering ===")
    df_engineered = preprocessor.feature_engineering(df_clean)
    
    # Encode categorical features
    print("\n=== Step 4: Encoding Categorical Features ===")
    categorical_features = ['gender', 'admission_type', 'discharge_disposition']
    df_encoded = preprocessor.encode_categorical_features(df_engineered, categorical_features)
    
    # Normalize numerical features
    print("\n=== Step 5: Normalizing Numerical Features ===")
    numerical_features = ['age', 'length_of_stay', 'num_procedures', 'num_medications']
    df_normalized = preprocessor.normalize_features(df_encoded, numerical_features)
    
    # Handle outliers
    print("\n=== Step 6: Handling Outliers ===")
    df_final = preprocessor.detect_and_handle_outliers(df_normalized, numerical_features)
    
    # Prepare train/validation/test splits
    print("\n=== Step 7: Creating Data Splits ===")
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.prepare_train_test_split(
        df_final, 
        target_col='readmitted_30days'
    )
    
    print("\n=== Preprocessing Complete ===")
    print(f"Final feature count: {X_train.shape[1]}")
    print(f"Ready for model training!")