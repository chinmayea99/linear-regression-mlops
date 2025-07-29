"""
Unit tests for the training pipeline
"""
import pytest
import numpy as np
import os
import joblib
from sklearn.linear_model import LinearRegression
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils import load_data, save_model_artifacts, load_model_artifacts
from train import train_model


class TestDataLoading:
    """Test data loading functionality"""
    
    def test_load_data_returns_correct_shapes(self):
        """Test that load_data returns data with correct shapes"""
        X_train, X_test, y_train, y_test, scaler = load_data()
        
        # Check that data is not empty
        assert X_train.shape[0] > 0
        assert X_test.shape[0] > 0
        assert y_train.shape[0] > 0
        assert y_test.shape[0] > 0
        
        # Check that train and test sets have same number of features
        assert X_train.shape[1] == X_test.shape[1]
        
        # Check that X and y have same number of samples
        assert X_train.shape[0] == y_train.shape[0]
        assert X_test.shape[0] == y_test.shape[0]
        
        # Check scaler is fitted
        assert hasattr(scaler, 'mean_')
        assert hasattr(scaler, 'scale_')
    
    def test_load_data_reproducible(self):
        """Test that load_data produces reproducible results"""
        X_train1, X_test1, y_train1, y_test1, _ = load_data(random_state=42)
        X_train2, X_test2, y_train2, y_test2, _ = load_data(random_state=42)
        
        np.testing.assert_array_equal(X_train1, X_train2)
        np.testing.assert_array_equal(X_test1, X_test2)
        np.testing.assert_array_equal(y_train1, y_train2)
        np.testing.assert_array_equal(y_test1, y_test2)


class TestModelTraining:
    """Test model training functionality"""
    
    def test_model_creation(self):
        """Test that LinearRegression model is created correctly"""
        X_train, _, y_train, _, _ = load_data()
        
        model = LinearRegression()
        assert isinstance(model, LinearRegression)
        
        # Train model
        model.fit(X_train, y_train)
        
        # Check that model has required attributes after training
        assert hasattr(model, 'coef_')
        assert hasattr(model, 'intercept_')
    
    def test_model_coefficients_exist(self):
        """Test that trained model has coefficients"""
        X_train, _, y_train, _, _ = load_data()
        
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Check coefficients exist and have correct shape
        assert model.coef_ is not None
        assert isinstance(model.coef_, np.ndarray)
        assert model.coef_.shape[0] == X_train.shape[1]
        
        # Check intercept exists
        assert model.intercept_ is not None
        assert isinstance(model.intercept_, (int, float, np.number))
    
    def test_model_r2_threshold(self):
        """Test that model achieves minimum R² score threshold"""
        model, _, r2_score = train_model()
        
        # Check that R² score exceeds minimum threshold (0.5 for this dataset)
        MIN_R2_THRESHOLD = 0.5
        assert r2_score > MIN_R2_THRESHOLD, f"R² score {r2_score:.4f} is below threshold {MIN_R2_THRESHOLD}"
        
        # Additional check that R² is reasonable (not too good to be true)
        assert r2_score < 1.0, f"R² score {r2_score:.4f} is suspiciously high"
    
    def test_model_predictions_shape(self):
        """Test that model predictions have correct shape"""
        X_train, X_test, y_train, y_test, _ = load_data()
        
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # Check prediction shapes
        assert y_pred_train.shape == y_train.shape
        assert y_pred_test.shape == y_test.shape
        
        # Check predictions are numeric
        assert np.all(np.isfinite(y_pred_train))
        assert np.all(np.isfinite(y_pred_test))


class TestModelPersistence:
    """Test model saving and loading"""
    
    def test_model_artifacts_saved(self):
        """Test that model artifacts are saved correctly"""
        # Train model
        model, scaler, _ = train_model()
        
        # Check that model files exist
        assert os.path.exists("models/linear_regression_model.joblib")
        assert os.path.exists("models/linear_regression_scaler.joblib")
        
        # Check that files are not empty
        assert os.path.getsize("models/linear_regression_model.joblib") > 0
        assert os.path.getsize("models/linear_regression_scaler.joblib") > 0
    
    def test_model_artifacts_loadable(self):
        """Test that saved model artifacts can be loaded"""
        # Ensure model is trained and saved
        train_model()
        
        # Load model artifacts
        loaded_model, loaded_scaler = load_model_artifacts()
        
        # Check that loaded objects are correct types
        assert isinstance(loaded_model, LinearRegression)
        assert hasattr(loaded_scaler, 'mean_')
        assert hasattr(loaded_scaler, 'scale_')
        
        # Check that loaded model has required attributes
        assert hasattr(loaded_model, 'coef_')
        assert hasattr(loaded_model, 'intercept_')
    
    def test_loaded_model_predictions(self):
        """Test that loaded model can make predictions"""
        # Ensure model is trained and saved
        train_model()
        
        # Load model and test data
        model, scaler = load_model_artifacts()
        _, X_test, _, y_test, _ = load_data()
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Check predictions
        assert y_pred.shape == y_test.shape
        assert np.all(np.isfinite(y_pred))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])