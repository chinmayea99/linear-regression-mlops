"""
Model prediction script for Docker container verification
Supports both original and quantized models
"""
import numpy as np
import joblib
import os
from utils import load_model_artifacts, load_data, calculate_metrics


def load_quantized_model():
    """
    Load and reconstruct quantized model
    """
    try:
        # Load quantized parameters
        quantized_params = joblib.load("models/quant_params.joblib")
        method = quantized_params['method']
        
        print(f"Loading quantized model (method: {method})...")
        
        if method == "symmetric":
            # Reconstruct from symmetric quantization
            quant_coef = quantized_params['quant_coef']
            coef_scale = quantized_params['coef_scale']
            intercept = quantized_params['intercept']
            
            # Dequantize coefficients
            dequant_coef = quant_coef.astype(np.float32) * coef_scale
            
            return dequant_coef, intercept, method
            
        elif method == "asymmetric":
            # Reconstruct from asymmetric quantization
            quant_coef = quantized_params['quant_coef']
            coef_scale = quantized_params['coef_scale']
            coef_zero_point = quantized_params['coef_zero_point']
            intercept = quantized_params['intercept']
            
            # Dequantize coefficients
            dequant_coef = coef_scale * (quant_coef.astype(np.float32) - coef_zero_point)
            
            return dequant_coef, intercept, method
            
        elif method == "16bit":
            # Reconstruct from 16-bit quantization
            quant_coef = quantized_params['quant_coef']
            coef_scale = quantized_params['coef_scale']
            intercept = quantized_params['intercept']
            
            # Dequantize coefficients
            dequant_coef = quant_coef.astype(np.float32) * coef_scale
            
            return dequant_coef, intercept, method
            
        else:
            raise ValueError(f"Unknown quantization method: {method}")
            
    except FileNotFoundError:
        print("Quantized model not found. Using original model.")
        return None, None, None


def run_predictions():
    """
    Run predictions using the trained model (original or quantized)
    """
    print("Loading trained model and test data...")
    
    try:
        # Load original model artifacts
        model, scaler = load_model_artifacts()
        
        # Load test data
        _, X_test, _, y_test, _ = load_data()
        
        print(f"Test data shape: {X_test.shape}")
        print(f"Model coefficients shape: {model.coef_.shape}")
        
        # Make predictions with original model
        print("\n=== Original Model Predictions ===")
        y_pred_original = model.predict(X_test)
        original_metrics = calculate_metrics(y_test, y_pred_original)
        
        print(f"Original Model R² Score: {original_metrics['r2_score']:.4f}")
        print(f"Original Model RMSE: {original_metrics['rmse']:.4f}")
        print(f"Original Model MAE: {original_metrics['mae']:.4f}")
        
        # Try to load and test quantized model
        quant_coef, quant_intercept, quant_method = load_quantized_model()
        
        if quant_coef is not None:
            print(f"\n=== Quantized Model Predictions ({quant_method}) ===")
            
            # Make predictions with quantized model
            y_pred_quantized = X_test @ quant_coef + quant_intercept
            quantized_metrics = calculate_metrics(y_test, y_pred_quantized)
            
            print(f"Quantized Model R² Score: {quantized_metrics['r2_score']:.4f}")
            print(f"Quantized Model RMSE: {quantized_metrics['rmse']:.4f}")
            print(f"Quantized Model MAE: {quantized_metrics['mae']:.4f}")
            
            # Compare models
            r2_diff = abs(original_metrics['r2_score'] - quantized_metrics['r2_score'])
            rmse_diff = abs(original_metrics['rmse'] - quantized_metrics['rmse'])
            
            print(f"\n=== Model Comparison ===")
            print(f"R² Score Difference: {r2_diff:.6f}")
            print(f"RMSE Difference: {rmse_diff:.6f}")
            print(f"Performance Retained: {(quantized_metrics['r2_score']/original_metrics['r2_score']*100):.2f}%")
            
            # Determine which predictions to show
            if quantized_metrics['r2_score'] > 0.5:
                print("\n✅ Using quantized model predictions (acceptable performance)")
                y_pred_final = y_pred_quantized
                final_metrics = quantized_metrics
                model_type = f"Quantized ({quant_method})"
            else:
                print("\n⚠️  Using original model predictions (quantized performance too low)")
                y_pred_final = y_pred_original
                final_metrics = original_metrics
                model_type = "Original"
        else:
            print("\n📝 Using original model predictions")
            y_pred_final = y_pred_original
            final_metrics = original_metrics
            model_type = "Original"
        
        # Show sample predictions
        print(f"\n=== Sample Predictions ({model_type} Model) ===")
        print("Index | True Value | Predicted | Absolute Error")
        print("-" * 50)
        
        for i in range(min(10, len(y_test))):
            abs_error = abs(y_test[i] - y_pred_final[i])
            print(f"{i:5d} | {y_test[i]:10.2f} | {y_pred_final[i]:9.2f} | {abs_error:13.3f}")
        
        # Summary statistics
        print(f"\n=== Final Model Performance ({model_type}) ===")
        print(f"R² Score: {final_metrics['r2_score']:.4f}")
        print(f"RMSE: {final_metrics['rmse']:.4f}")
        print(f"MAE: {final_metrics['mae']:.4f}")
        print(f"MSE: {final_metrics['mse']:.4f}")
        
        # Validation check
        if final_metrics['r2_score'] > 0.5:
            print(f"\n✅ Model performance is acceptable (R² > 0.5)")
            print("Prediction completed successfully!")
            return True
        else:
            print(f"\n❌ Warning: Model performance is below threshold (R² ≤ 0.5)")
            print("This may indicate a problem with the model or data.")
            return False
        
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_predictions()
    if not success:
        exit(1)