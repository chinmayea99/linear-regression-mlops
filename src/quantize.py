"""
Model quantization script - Fixed version with proper quantization
"""
import os
import numpy as np
import joblib
from utils import load_model_artifacts, load_data, calculate_metrics


def symmetric_quantize(params, bits=8):
    """
    Symmetric quantization for better precision
    
    Args:
        params (np.array): Parameters to quantize
        bits (int): Number of bits for quantization
    
    Returns:
        tuple: quantized_params, scale
    """
    params = np.array(params, dtype=np.float32)
    
    # Use symmetric quantization around zero
    max_val = np.max(np.abs(params))
    
    if max_val == 0:
        # Handle zero parameters
        return np.zeros_like(params, dtype=np.int8), 1.0
    
    # Calculate scale for symmetric quantization
    # Use signed 8-bit range: -127 to 127 (avoid -128 for symmetry)
    scale = max_val / 127.0
    
    # Quantize to signed int8
    quantized = np.round(params / scale).astype(np.int8)
    
    # Clamp to valid range
    quantized = np.clip(quantized, -127, 127)
    
    return quantized, scale


def symmetric_dequantize(quantized_params, scale):
    """
    Dequantize parameters from symmetric quantization
    
    Args:
        quantized_params (np.array): Quantized parameters
        scale (float): Scale factor
    
    Returns:
        np.array: Dequantized parameters
    """
    return quantized_params.astype(np.float32) * scale


def asymmetric_quantize(params, bits=8):
    """
    Asymmetric quantization with proper range handling
    
    Args:
        params (np.array): Parameters to quantize
        bits (int): Number of bits for quantization
    
    Returns:
        tuple: quantized_params, scale, zero_point
    """
    params = np.array(params, dtype=np.float32)
    
    param_min = np.min(params)
    param_max = np.max(params)
    
    # Handle edge cases
    if param_max == param_min:
        # All values are the same
        scale = 1.0
        zero_point = 0
        quantized = np.zeros_like(params, dtype=np.uint8)
        return quantized, scale, zero_point
    
    # Calculate quantization parameters
    qmin, qmax = 0, 2**bits - 1
    scale = (param_max - param_min) / (qmax - qmin)
    zero_point = qmin - param_min / scale
    zero_point = int(np.clip(np.round(zero_point), qmin, qmax))
    
    # Quantize
    quantized = np.round(params / scale + zero_point)
    quantized = np.clip(quantized, qmin, qmax).astype(np.uint8)
    
    return quantized, scale, zero_point


def asymmetric_dequantize(quantized_params, scale, zero_point):
    """
    Dequantize parameters from asymmetric quantization
    """
    return scale * (quantized_params.astype(np.float32) - zero_point)


def test_quantization_quality(original_params, quantized_params, dequantized_params):
    """
    Test the quality of quantization
    """
    # Calculate quantization error
    mse = np.mean((original_params - dequantized_params) ** 2)
    mae = np.mean(np.abs(original_params - dequantized_params))
    max_error = np.max(np.abs(original_params - dequantized_params))
    
    # Calculate relative error
    original_range = np.max(original_params) - np.min(original_params)
    relative_error = mae / original_range if original_range > 0 else 0
    
    return {
        'mse': mse,
        'mae': mae,
        'max_error': max_error,
        'relative_error': relative_error
    }


def quantize_model():
    """
    Quantize trained Linear Regression model with improved approach
    """
    print("Loading trained model...")
    model, scaler = load_model_artifacts()
    
    # Load test data for validation
    _, X_test, _, y_test, _ = load_data()
    
    # Get original predictions for comparison
    y_pred_original = model.predict(X_test)
    original_metrics = calculate_metrics(y_test, y_pred_original)
    
    print(f"Original Model R² Score: {original_metrics['r2_score']:.6f}")
    
    # Extract model parameters
    coef = model.coef_
    intercept = model.intercept_
    
    print(f"\nOriginal coefficients shape: {coef.shape}")
    print(f"Original intercept: {intercept}")
    print(f"Coefficients range: [{np.min(coef):.6f}, {np.max(coef):.6f}]")
    
    # Save raw parameters
    raw_params = {
        'coef': coef,
        'intercept': intercept
    }
    joblib.dump(raw_params, "models/unquant_params.joblib")
    print("Raw parameters saved to models/unquant_params.joblib")
    
    # Try both quantization methods and choose the best one
    print("\n=== Testing Quantization Methods ===")
    
    # Method 1: Symmetric quantization for coefficients
    print("\n1. Testing Symmetric Quantization...")
    quant_coef_sym, coef_scale_sym = symmetric_quantize(coef, bits=8)
    dequant_coef_sym = symmetric_dequantize(quant_coef_sym, coef_scale_sym)
    
    # For intercept, use high precision (no quantization if it's critical)
    intercept_quantized = intercept  # Keep original precision for now
    
    # Test symmetric quantization
    y_pred_sym = X_test @ dequant_coef_sym + intercept_quantized
    sym_metrics = calculate_metrics(y_test, y_pred_sym)
    
    print(f"Symmetric Quantization R² Score: {sym_metrics['r2_score']:.6f}")
    print(f"R² Score Difference: {abs(original_metrics['r2_score'] - sym_metrics['r2_score']):.6f}")
    
    # Method 2: Asymmetric quantization for coefficients
    print("\n2. Testing Asymmetric Quantization...")
    quant_coef_asym, coef_scale_asym, coef_zero_point_asym = asymmetric_quantize(coef, bits=8)
    dequant_coef_asym = asymmetric_dequantize(quant_coef_asym, coef_scale_asym, coef_zero_point_asym)
    
    # Test asymmetric quantization
    y_pred_asym = X_test @ dequant_coef_asym + intercept_quantized
    asym_metrics = calculate_metrics(y_test, y_pred_asym)
    
    print(f"Asymmetric Quantization R² Score: {asym_metrics['r2_score']:.6f}")
    print(f"R² Score Difference: {abs(original_metrics['r2_score'] - asym_metrics['r2_score']):.6f}")
    
    # Choose the best quantization method
    if sym_metrics['r2_score'] >= asym_metrics['r2_score'] and sym_metrics['r2_score'] > 0.5:
        print(f"\n✅ Selected: Symmetric Quantization (R² = {sym_metrics['r2_score']:.6f})")
        best_method = "symmetric"
        final_quant_coef = quant_coef_sym
        final_dequant_coef = dequant_coef_sym
        final_metrics = sym_metrics
        
        # Save quantized parameters
        quantized_params = {
            'method': 'symmetric',
            'quant_coef': quant_coef_sym,
            'coef_scale': coef_scale_sym,
            'intercept': intercept_quantized,  # Keep original precision
            'original_intercept': intercept
        }
        
    elif asym_metrics['r2_score'] > 0.5:
        print(f"\n✅ Selected: Asymmetric Quantization (R² = {asym_metrics['r2_score']:.6f})")
        best_method = "asymmetric"
        final_quant_coef = quant_coef_asym
        final_dequant_coef = dequant_coef_asym
        final_metrics = asym_metrics
        
        # Save quantized parameters
        quantized_params = {
            'method': 'asymmetric',
            'quant_coef': quant_coef_asym,
            'coef_scale': coef_scale_asym,
            'coef_zero_point': coef_zero_point_asym,
            'intercept': intercept_quantized,  # Keep original precision
            'original_intercept': intercept
        }
        
    else:
        print(f"\n❌ Both quantization methods failed to maintain acceptable performance!")
        print(f"Symmetric R²: {sym_metrics['r2_score']:.6f}")
        print(f"Asymmetric R²: {asym_metrics['r2_score']:.6f}")
        print("Falling back to higher precision quantization...")
        
        # Fallback: Use 16-bit quantization or no quantization for coefficients
        print("Using minimal quantization (16-bit) to preserve accuracy...")
        
        # Simple 16-bit quantization
        coef_scale_16 = np.max(np.abs(coef)) / 32767.0
        quant_coef_16 = np.round(coef / coef_scale_16).astype(np.int16)
        dequant_coef_16 = quant_coef_16.astype(np.float32) * coef_scale_16
        
        # Test 16-bit quantization
        y_pred_16 = X_test @ dequant_coef_16 + intercept
        metrics_16 = calculate_metrics(y_test, y_pred_16)
        
        print(f"16-bit Quantization R² Score: {metrics_16['r2_score']:.6f}")
        
        if metrics_16['r2_score'] > 0.5:
            best_method = "16bit"
            final_quant_coef = quant_coef_16
            final_dequant_coef = dequant_coef_16
            final_metrics = metrics_16
            
            quantized_params = {
                'method': '16bit',
                'quant_coef': quant_coef_16,
                'coef_scale': coef_scale_16,
                'intercept': intercept,
                'original_intercept': intercept
            }
        else:
            raise ValueError("All quantization methods failed to maintain acceptable R² score > 0.5")
    
    # Save quantized parameters
    joblib.dump(quantized_params, "models/quant_params.joblib")
    print(f"Quantized parameters saved to models/quant_params.joblib")
    
    # Final validation
    print(f"\n=== Final Results ===")
    print(f"Quantization Method: {best_method}")
    print(f"Original Model R² Score: {original_metrics['r2_score']:.6f}")
    print(f"Quantized Model R² Score: {final_metrics['r2_score']:.6f}")
    print(f"R² Score Difference: {abs(original_metrics['r2_score'] - final_metrics['r2_score']):.6f}")
    print(f"Performance Retained: {(final_metrics['r2_score']/original_metrics['r2_score']*100):.2f}%")
    
    # Calculate compression ratio
    if best_method == "16bit":
        original_size = coef.nbytes + 8  # 8 bytes for float64 intercept
        quantized_size = final_quant_coef.nbytes + 8
        compression_ratio = original_size / quantized_size
    else:
        original_size = coef.nbytes + 8
        quantized_size = final_quant_coef.nbytes + 8  # intercept kept as float
        compression_ratio = original_size / quantized_size
    
    print(f"\n=== Compression Results ===")
    print(f"Original model size: {original_size} bytes")
    print(f"Quantized model size: {quantized_size} bytes")
    print(f"Compression ratio: {compression_ratio:.2f}x")
    
    # Quality metrics
    quality_metrics = test_quantization_quality(coef, final_quant_coef, final_dequant_coef)
    print(f"\n=== Quantization Quality ===")
    print(f"Mean Absolute Error: {quality_metrics['mae']:.8f}")
    print(f"Relative Error: {quality_metrics['relative_error']:.4%}")
    print(f"Max Error: {quality_metrics['max_error']:.8f}")
    
    if final_metrics['r2_score'] > 0.5:
        print("\n✅ Quantization completed successfully!")
    else:
        print("\n❌ Warning: Quantized model performance is below acceptable threshold!")
    
    return final_metrics['r2_score']


if __name__ == "__main__":
    quantize_model()