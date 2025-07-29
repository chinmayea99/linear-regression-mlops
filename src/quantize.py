"""
Model quantization script
"""
import os
import numpy as np
import joblib
from utils import load_model_artifacts, load_data, calculate_metrics


def quantize_parameters(params, bits=8):
    # Calculate quantization parameters
    param_min = np.min(params)
    param_max = np.max(params)
    
    # Handle edge case where min == max (single value or all values are same)
    if param_max == param_min:
        # For constant values, use a small scale to avoid division by zero
        scale = 1e-8
        zero_point = 0
        # Quantize to middle value (127 for 8-bit)
        quantized = np.full_like(params, 127, dtype=np.uint8)
    else:
        # Calculate scale and zero point normally
        scale = (param_max - param_min) / (2**bits - 1)
        zero_point = int(np.clip(-param_min / scale, 0, 2**bits - 1))
        
        # Quantize
        quantized_float = (params / scale) + zero_point
        quantized = np.clip(np.round(quantized_float), 0, 2**bits - 1).astype(np.uint8)
    
    return quantized, scale, zero_point


def dequantize_parameters(quantized_params, scale, zero_point):
    # Handle edge case where scale is very small (constant value case)
    if scale < 1e-6:
        # Return the original constant value
        return quantized_params.astype(np.float32) * 0 + (quantized_params[0] - zero_point) * scale
    else:
        return scale * (quantized_params.astype(np.float32) - zero_point)


def quantize_model():
    """
    Quantize trained Linear Regression model
    """
    print("Loading trained model...")
    model, scaler = load_model_artifacts()
    
    # Extract model parameters
    coef = model.coef_
    intercept = model.intercept_
    
    print(f"Original coefficients shape: {coef.shape}")
    print(f"Original intercept: {intercept}")
    
    # Save raw parameters
    raw_params = {
        'coef': coef,
        'intercept': intercept
    }
    joblib.dump(raw_params, "models/unquant_params.joblib")
    print("Raw parameters saved to models/unquant_params.joblib")
    
    # Quantize coefficients
    print("\nQuantizing coefficients...")
    quant_coef, coef_scale, coef_zero_point = quantize_parameters(coef)
    
    # For intercept, we'll use a different approach to handle single values better
    print("Quantizing intercept...")
    
    # Create a small range around the intercept for better quantization
    intercept_range = max(abs(intercept) * 0.001, 1e-6)  # 0.1% of intercept or minimum threshold
    intercept_array = np.array([intercept - intercept_range, intercept, intercept + intercept_range])
    
    # Quantize the range and extract the middle value
    quant_intercept_array, intercept_scale, intercept_zero_point = quantize_parameters(intercept_array)
    quant_intercept = np.array([quant_intercept_array[1]], dtype=np.uint8)  # Take middle value
    
    # Adjust the scale and zero_point for the actual intercept value
    intercept_scale = intercept_scale
    intercept_zero_point = intercept_zero_point
    
    print(f"Intercept quantization - Scale: {intercept_scale:.8f}, Zero Point: {intercept_zero_point}")
    
    # Save quantized parameters
    quantized_params = {
        'quant_coef': quant_coef,
        'coef_scale': coef_scale,
        'coef_zero_point': coef_zero_point,
        'quant_intercept': quant_intercept,
        'intercept_scale': intercept_scale,
        'intercept_zero_point': intercept_zero_point,
        'original_intercept': intercept  # Store original for verification
    }
    joblib.dump(quantized_params, "models/quant_params.joblib")
    print("Quantized parameters saved to models/quant_params.joblib")
    
    # Test dequantization and inference
    print("\n=== Testing Quantized Model ===")
    
    # Dequantize parameters
    dequant_coef = dequantize_parameters(quant_coef, coef_scale, coef_zero_point)
    dequant_intercept = dequantize_parameters(quant_intercept, intercept_scale, intercept_zero_point)[0]
    
    print(f"Dequantized coefficients shape: {dequant_coef.shape}")
    print(f"Dequantized intercept: {dequant_intercept}")
    
    # Compare original vs dequantized parameters
    coef_diff = np.mean(np.abs(coef - dequant_coef))
    intercept_diff = abs(intercept - dequant_intercept)
    
    print(f"\nParameter Differences:")
    print(f"Mean absolute difference in coefficients: {coef_diff:.6f}")
    print(f"Absolute difference in intercept: {intercept_diff:.6f}")
    
    # Test inference with dequantized model
    print("\nTesting inference with dequantized model...")
    X_train, X_test, y_train, y_test, _ = load_data()
    
    # Original model predictions
    y_pred_original = model.predict(X_test)
    
    # Dequantized model predictions (manual calculation)
    y_pred_dequant = X_test @ dequant_coef + dequant_intercept
    
    # Calculate metrics for both
    original_metrics = calculate_metrics(y_test, y_pred_original)
    dequant_metrics = calculate_metrics(y_test, y_pred_dequant)
    
    print(f"\nOriginal Model R² Score: {original_metrics['r2_score']:.6f}")
    print(f"Dequantized Model R² Score: {dequant_metrics['r2_score']:.6f}")
    print(f"R² Score Difference: {abs(original_metrics['r2_score'] - dequant_metrics['r2_score']):.6f}")
    
    # Calculate compression ratio
    original_size = (coef.nbytes + np.array([intercept]).nbytes)
    quantized_size = (quant_coef.nbytes + quant_intercept.nbytes)
    compression_ratio = original_size / quantized_size
    
    print(f"\n=== Compression Results ===")
    print(f"Original model size: {original_size} bytes")
    print(f"Quantized model size: {quantized_size} bytes")
    print(f"Compression ratio: {compression_ratio:.2f}x")
    
    print("\nQuantization completed successfully!")


if __name__ == "__main__":
    quantize_model()