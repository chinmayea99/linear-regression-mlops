"""
Model prediction script for Docker container verification
"""
import numpy as np
from utils import load_model_artifacts, load_data, calculate_metrics


def run_predictions():
    """
    Run predictions using the trained model
    """
    print("Loading trained model and test data...")
    
    try:
        # Load model artifacts
        model, scaler = load_model_artifacts()
        
        # Load test data
        _, X_test, _, y_test, _ = load_data()
        
        print(f"Test data shape: {X_test.shape}")
        print(f"Model coefficients shape: {model.coef_.shape}")
        
        # Make predictions
        print("\nRunning predictions...")
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        metrics = calculate_metrics(y_test, y_pred)
        
        # Print results
        print("\n=== Prediction Results ===")
        print(f"R² Score: {metrics['r2_score']:.4f}")
        print(f"RMSE: {metrics['rmse']:.4f}")
        print(f"MAE: {metrics['mae']:.4f}")
        
        # Show sample predictions
        print("\n=== Sample Predictions ===")
        for i in range(min(10, len(y_test))):
            print(f"True: {y_test[i]:.2f}, Predicted: {y_pred[i]:.2f}, Diff: {abs(y_test[i] - y_pred[i]):.2f}")
        
        print("\nPrediction completed successfully!")
        return True
        
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return False


if __name__ == "__main__":
    success = run_predictions()
    if not success:
        exit(1)