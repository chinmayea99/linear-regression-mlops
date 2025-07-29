"""
Model training script
"""
import os
import numpy as np
from sklearn.linear_model import LinearRegression
from utils import load_data, save_model_artifacts, calculate_metrics


def train_model():
    """
    Train Linear Regression model on California Housing dataset
    """
    print("Loading California Housing dataset...")
    X_train, X_test, y_train, y_test, scaler = load_data()
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    
    # Initialize and train model
    print("Training Linear Regression model...")
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Make predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Calculate metrics
    train_metrics = calculate_metrics(y_train, y_train_pred)
    test_metrics = calculate_metrics(y_test, y_test_pred)
    
    # Print results
    print("\n=== Training Results ===")
    print(f"Training R² Score: {train_metrics['r2_score']:.4f}")
    print(f"Training RMSE: {train_metrics['rmse']:.4f}")
    print(f"Training MAE: {train_metrics['mae']:.4f}")
    
    print(f"\nTest R² Score: {test_metrics['r2_score']:.4f}")
    print(f"Test RMSE: {test_metrics['rmse']:.4f}")
    print(f"Test MAE: {test_metrics['mae']:.4f}")
    
    # Print model parameters
    print(f"\nModel Coefficients Shape: {model.coef_.shape}")
    print(f"Model Intercept: {model.intercept_:.4f}")
    
    # Save model artifacts
    save_model_artifacts(model, scaler)
    
    print("\nModel training completed successfully!")
    
    return model, scaler, test_metrics['r2_score']


if __name__ == "__main__":
    train_model()