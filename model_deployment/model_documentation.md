# Heart Disease Prediction Model - Documentation

## Model Overview
This is a trained machine learning model for predicting heart disease based on patient medical data.

**Generated on:** 2025-09-25 11:37:03
**Version:** 1.0
**Model Type:** Optimized Classification Model

## Files Structure
```
model_deployment/
├── heart_disease_predictor_pipeline.pkl    # Main pipeline (use this)
├── heart_disease_predictor_pipeline.pickle # Pickle backup
├── model_metadata.json                     # Model information
├── deployment_example.py                   # Usage examples
├── requirements.txt                        # Python dependencies
├── model_documentation.md                  # This file
└── components/                             # Individual components
    ├── model.pkl                          # Trained model
    └── scaler.pkl                         # Feature scaler
```

## Quick Start

### 1. Installation
```python
pip install -r requirements.txt
```

### 2. Load Model
```python
import joblib
predictor = joblib.load('heart_disease_predictor_pipeline.pkl')
```

### 3. Make Predictions
```python
import pandas as pd

# Prepare your data as DataFrame
patient_data = pd.DataFrame([{
    # Your feature values here
}])

# Get prediction
result = predictor.get_prediction_with_confidence(patient_data)[0]
print(f"Diagnosis: {result['diagnosis']}")
print(f"Confidence: {result['confidence']:.3f}")
```

## Model Features
The model expects the following input features:
- All features should be numeric
- Missing values should be handled before prediction
- Features are automatically scaled by the pipeline

## API Reference

### HeartDiseasePredictor Class

#### Methods:
- `predict(X)`: Returns binary predictions (0/1)
- `predict_proba(X)`: Returns prediction probabilities 
- `get_prediction_with_confidence(X)`: Returns detailed predictions with confidence
- `get_model_info()`: Returns model metadata

#### Prediction Output Format:
```python
{
    'prediction': 0 or 1,           # 0 = No Disease, 1 = Disease
    'diagnosis': str,               # Human readable diagnosis
    'confidence': float,            # Confidence score (0-1)
    'risk_level': str,             # 'Low', 'Medium', or 'High'
    'probabilities': [float, float] # [prob_no_disease, prob_disease]
}
```

## Model Performance
The model has been validated and optimized using hyperparameter tuning.
Check `model_metadata.json` for detailed performance metrics.

## Troubleshooting

### Common Issues:
1. **Feature Mismatch**: Ensure input features match training features
2. **Data Types**: All features should be numeric
3. **Missing Values**: Handle missing data before prediction
4. **Scaling**: Don't manually scale - the pipeline handles this

### Support:
- Check `deployment_example.py` for usage examples
- Review `model_metadata.json` for model details
- Ensure all requirements are installed

## Model Maintenance
- Retrain model with new data periodically
- Monitor prediction performance in production
- Update model version as needed

## Deployment Notes
- Model is thread-safe for concurrent predictions
- Consider caching for high-frequency predictions
- Monitor memory usage for batch predictions
- Log predictions for model monitoring
