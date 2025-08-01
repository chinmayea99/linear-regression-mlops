# Linear Regression MLOps Pipeline

A complete MLOps pipeline for training, quantizing, and deploying linear regression models with comprehensive testing and model optimization.

## 🎯 Project Overview

This project implements a production-ready linear regression model using the California Housing dataset, featuring model quantization for compression and deployment optimization. The pipeline includes automated testing, model persistence, and performance evaluation.

## 📊 Model Performance

### Training Results
- **Training R² Score**: 0.6126
- **Training RMSE**: 0.7197
- **Training MAE**: 0.5286
- **Test R² Score**: 0.5758
- **Test RMSE**: 0.7456
- **Test MAE**: 0.5332

### Quantization Results
- **Original Model R² Score**: 0.5758
- **Quantized Model R² Score**: 0.5767
- **Performance Improvement**: +0.16%
- **Compression Ratio**: 4.50x (72 bytes → 16 bytes)
- **Quantization Error**: 0.1093% relative error

## 🏗️ Project Structure

```
linear-regression-mlops/
├── src/
│   ├── train.py          # Model training script
│   ├── quantize.py       # Model quantization
│   └── predict.py        # Prediction and evaluation
├── tests/
│   └── test_train.py     # Comprehensive test suite
├── models/               # Saved model artifacts
└── README.md
```

## 🚀 Features

### Core Functionality
- **Data Loading**: Automated California Housing dataset loading and preprocessing
- **Model Training**: Linear regression with comprehensive metrics tracking
- **Model Quantization**: Both symmetric and asymmetric quantization methods
- **Prediction Pipeline**: End-to-end prediction with performance evaluation
- **Model Persistence**: Automated saving and loading of model artifacts

### Quantization Methods
1. **Symmetric Quantization**: R² Score = 0.576287
2. **Asymmetric Quantization**: R² Score = 0.576725 ✅ (Selected)

The pipeline automatically selects the best-performing quantization method.

## 📈 Dataset Information

- **Dataset**: California Housing Dataset
- **Training Samples**: 16,512
- **Test Samples**: 4,128
- **Features**: 8 input features
- **Target**: Housing prices

## 🧪 Testing Suite

Comprehensive test coverage with 9 test cases:

### Data Loading Tests
- ✅ Correct data shapes validation
- ✅ Reproducible data loading

### Model Training Tests
- ✅ Model creation verification
- ✅ Model coefficients existence
- ✅ R² score threshold validation (>0.5)
- ✅ Prediction shape consistency

### Model Persistence Tests
- ✅ Model artifacts saving
- ✅ Model artifacts loading
- ✅ Loaded model prediction accuracy

**Test Results**: 9/9 tests passed in 3.67s

## 🔧 Installation & Usage

### Prerequisites
```bash
pip install scikit-learn numpy joblib pytest
```

### Running the Pipeline

1. **Train the Model**
```bash
python src/train.py
```

2. **Quantize the Model**
```bash
python src/quantize.py
```

3. **Make Predictions**
```bash
python src/predict.py
```

4. **Run Tests**
```bash
python -m pytest tests/ -v
```

## 📋 Performance Metrics

### Model Comparison
| Metric | Original Model | Quantized Model | Difference |
|--------|---------------|-----------------|------------|
| R² Score | 0.5758 | 0.5767 | +0.0009 |
| RMSE | 0.7456 | 0.7448 | -0.0008 |
| MAE | 0.5332 | 0.5335 | +0.0003 |
| Model Size | 72 bytes | 16 bytes | -78% |

### Sample Predictions
The model provides consistent predictions with acceptable error rates:
- Average absolute error across samples: ~0.6
- Model maintains good generalization performance

## 🎖️ Key Achievements

- **Model Compression**: 4.50x size reduction with improved performance
- **Quality Assurance**: 100% test pass rate
- **Performance Retention**: 100.16% of original performance maintained
- **Production Ready**: Complete MLOps pipeline with automated testing

## 🔍 Model Quality Metrics

### Quantization Quality Assessment
- **Mean Absolute Error**: 0.00191418
- **Relative Error**: 0.1093%
- **Max Error**: 0.00276473

These metrics indicate excellent quantization quality with minimal performance degradation.

## 🚦 Model Validation

The model meets production readiness criteria:
- ✅ R² Score > 0.5 (actual: 0.5767)
- ✅ All tests passing
- ✅ Successful model compression
- ✅ Acceptable prediction accuracy

## 📝 Future Enhancements

- Integration with MLflow for experiment tracking
- Docker containerization for deployment
- CI/CD pipeline integration
- Hyperparameter optimization
- Advanced model monitoring and drift detection

## 🤝 Contributing

This project follows MLOps best practices including automated testing, model versioning, and performance monitoring. Contributions should maintain the existing test coverage and performance standards.



