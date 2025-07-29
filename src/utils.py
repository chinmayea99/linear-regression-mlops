"""
Utility functions for the MLOps pipeline
"""
import os
import joblib
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data(test_size=0.2, random_state=42):
    # Load dataset
    housing = fetch_california_housing()
    X, y = housing.data, housing.target
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def save_model_artifacts(model, scaler, model_path="models/", model_name="linear_regression"):
    os.makedirs(model_path, exist_ok=True)
    
    # Save model
    joblib.dump(model, f"{model_path}{model_name}_model.joblib")
    
    # Save scaler
    joblib.dump(scaler, f"{model_path}{model_name}_scaler.joblib")
    
    print(f"Model artifacts saved to {model_path}")


def load_model_artifacts(model_path="models/", model_name="linear_regression"):
    model = joblib.load(f"{model_path}{model_name}_model.joblib")
    scaler = joblib.load(f"{model_path}{model_name}_scaler.joblib")
    
    return model, scaler


def calculate_metrics(y_true, y_pred):
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    
    metrics = {
        'r2_score': r2_score(y_true, y_pred),
        'mse': mean_squared_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred)
    }
    
    return metrics