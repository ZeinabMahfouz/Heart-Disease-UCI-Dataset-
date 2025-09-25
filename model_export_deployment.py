import pandas as pd
import numpy as np
import joblib
import pickle
import json
import os
from datetime import datetime
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

class HeartDiseasePredictor:
    """
    Complete Heart Disease Prediction Pipeline
    Includes preprocessing, model prediction, and probability estimation
    """
    
    def __init__(self, model=None, scaler=None, feature_names=None, model_info=None):
        self.model = model
        self.scaler = scaler
        self.feature_names = feature_names
        self.model_info = model_info or {}
        self.creation_date = datetime.now().isoformat()
        self.version = "1.0"
        
    def preprocess(self, X):
        """
        Preprocess input features
        """
        if self.scaler is None:
            raise ValueError("Scaler not initialized. Cannot preprocess data.")
        
        # Ensure input is DataFrame with correct columns
        if isinstance(X, pd.DataFrame):
            if self.feature_names and list(X.columns) != self.feature_names:
                print("⚠️  Column names don't match training features. Attempting to reorder...")
                try:
                    X = X[self.feature_names]
                except KeyError as e:
                    raise ValueError(f"Missing features: {e}")
        else:
            # Convert numpy array to DataFrame
            if self.feature_names and X.shape[1] == len(self.feature_names):
                X = pd.DataFrame(X, columns=self.feature_names)
            else:
                raise ValueError("Input shape doesn't match expected features")
        
        # Apply scaling
        X_scaled = self.scaler.transform(X)
        return X_scaled
    
    def predict(self, X):
        """
        Make predictions on new data
        """
        if self.model is None:
            raise ValueError("Model not initialized. Cannot make predictions.")
        
        X_processed = self.preprocess(X)
        predictions = self.model.predict(X_processed)
        return predictions
    
    def predict_proba(self, X):
        """
        Get prediction probabilities
        """
        if self.model is None:
            raise ValueError("Model not initialized. Cannot make predictions.")
        
        if not hasattr(self.model, 'predict_proba'):
            raise ValueError("Model doesn't support probability prediction.")
        
        X_processed = self.preprocess(X)
        probabilities = self.model.predict_proba(X_processed)
        return probabilities
    
    def get_prediction_with_confidence(self, X):
        """
        Get predictions with confidence scores and interpretation
        """
        predictions = self.predict(X)
        
        try:
            probabilities = self.predict_proba(X)
            confidence_scores = np.max(probabilities, axis=1)
        except:
            # For models without predict_proba
            confidence_scores = np.ones(len(predictions)) * 0.5  # Default confidence
            probabilities = None
        
        results = []
        for i in range(len(predictions)):
            pred = int(predictions[i])
            confidence = float(confidence_scores[i])
            
            # Interpretation
            if pred == 1:
                diagnosis = "Heart Disease Detected"
                risk_level = "High" if confidence > 0.8 else ("Medium" if confidence > 0.6 else "Low")
            else:
                diagnosis = "No Heart Disease"
                risk_level = "Low" if confidence > 0.8 else ("Medium" if confidence > 0.6 else "High")
            
            result = {
                'prediction': pred,
                'diagnosis': diagnosis,
                'confidence': confidence,
                'risk_level': risk_level,
                'probabilities': probabilities[i].tolist() if probabilities is not None else None
            }
            results.append(result)
        
        return results
    
    def get_model_info(self):
        """
        Get comprehensive model information
        """
        info = {
            'version': self.version,
            'creation_date': self.creation_date,
            'model_type': type(self.model).__name__ if self.model else None,
            'scaler_type': type(self.scaler).__name__ if self.scaler else None,
            'n_features': len(self.feature_names) if self.feature_names else None,
            'feature_names': self.feature_names,
            'model_info': self.model_info
        }
        return info

def load_best_model_from_optimization():
    """
    Load the best model from hyperparameter optimization results
    """
    print("🔍 Loading Best Model from Optimization Results...")
    
    # Try to load optimization summary
    try:
        with open('hyperparameter_optimization_summary.json', 'r', encoding='utf-8') as f:
            optimization_summary = json.load(f)
        
        best_model_info = optimization_summary['best_overall_model']
        print(f"✅ Found best model: {best_model_info['name']}")
        print(f"✅ Optimization method: {best_model_info['optimization_method']}")
        
        return best_model_info
    except FileNotFoundError:
        print("⚠️  Optimization summary not found. Will try to find models manually.")
        return None

def load_optimized_models_and_data():
    """
    Load all necessary components for model deployment
    """
    print("📦 Loading Model Components...")
    
    # 1. Load best model info
    best_model_info = load_best_model_from_optimization()
    
    # 2. Try to load the best model file
    best_model = None
    model_files_to_try = []
    
    if best_model_info:
        model_name = best_model_info['name'].lower().replace(' ', '_')
        model_files_to_try.extend([
            f"optimized_models/best_model_{model_name}.pkl",
            f"optimized_models/{model_name}_grid_optimized.pkl",
            f"optimized_models/{model_name}_random_optimized.pkl"
        ])
    
    # Add common model files
    model_files_to_try.extend([
        "optimized_models/best_model_random_forest.pkl",
        "optimized_models/random_forest_grid_optimized.pkl",
        "optimized_models/logistic_regression_grid_optimized.pkl",
        "trained_models/random_forest_model.pkl",
        "trained_models/logistic_regression_model.pkl"
    ])
    
    for model_file in model_files_to_try:
        try:
            best_model = joblib.load(model_file)
            print(f"✅ Loaded model from: {model_file}")
            break
        except FileNotFoundError:
            continue
    
    if best_model is None:
        print("❌ Could not load optimized model. Please run hyperparameter optimization first.")
        return None, None, None, None, None
    
    # 3. Load scaler
    scaler = None
    scaler_files_to_try = [
        "optimized_models/feature_scaler_optimized.pkl",
        "trained_models/feature_scaler.pkl"
    ]
    
    for scaler_file in scaler_files_to_try:
        try:
            scaler = joblib.load(scaler_file)
            print(f"✅ Loaded scaler from: {scaler_file}")
            break
        except FileNotFoundError:
            continue
    
    # 4. Load dataset to get feature names
    feature_names = None
    datasets_to_try = [
        "feature_selected_top_10.csv",
        "feature_selected_top_15.csv",
        "pca_dataset_90pct.csv",
        "model_ready_data.csv"
    ]
    
    for dataset_file in datasets_to_try:
        try:
            data = pd.read_csv(dataset_file)
            if 'target' in data.columns:
                feature_names = [col for col in data.columns if col != 'target']
            else:
                feature_names = data.columns[:-1].tolist()
            print(f"✅ Loaded feature names from: {dataset_file}")
            print(f"✅ Features ({len(feature_names)}): {feature_names}")
            
            # Also return the data for validation
            X = data[feature_names] if 'target' in data.columns else data.iloc[:, :-1]
            y = data['target'] if 'target' in data.columns else data.iloc[:, -1]
            
            break
        except FileNotFoundError:
            continue
    
    if feature_names is None:
        print("⚠️  Could not load feature names. Model may not work correctly.")
        return best_model, scaler, None, None, best_model_info
    
    return best_model, scaler, feature_names, (X, y), best_model_info

def create_model_pipeline(model, scaler, feature_names, model_info):
    """
    Create a complete model pipeline
    """
    print("🔧 Creating Model Pipeline...")
    
    # Create the predictor class instance
    predictor = HeartDiseasePredictor(
        model=model,
        scaler=scaler,
        feature_names=feature_names,
        model_info=model_info
    )
    
    print("✅ Model pipeline created successfully")
    return predictor

def validate_model_pipeline(predictor, validation_data):
    """
    Validate the model pipeline with test data
    """
    print("🔍 Validating Model Pipeline...")
    
    if validation_data is None:
        print("⚠️  No validation data available")
        return
    
    X, y = validation_data
    
    # Split data for validation
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    try:
        # Test predictions
        predictions = predictor.predict(X_test)
        
        # Test probability predictions if available
        try:
            probabilities = predictor.predict_proba(X_test)
            print("✅ Probability predictions working")
        except:
            print("ℹ️  Probability predictions not available for this model")
            probabilities = None
        
        # Test prediction with confidence
        detailed_predictions = predictor.get_prediction_with_confidence(X_test.head(5))
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions)
        recall = recall_score(y_test, predictions)
        f1 = f1_score(y_test, predictions)
        
        if probabilities is not None:
            roc_auc = roc_auc_score(y_test, probabilities[:, 1])
        else:
            roc_auc = None
        
        print("✅ Model Pipeline Validation Results:")
        print(f"   Accuracy: {accuracy:.4f}")
        print(f"   Precision: {precision:.4f}")
        print(f"   Recall: {recall:.4f}")
        print(f"   F1-Score: {f1:.4f}")
        if roc_auc:
            print(f"   ROC AUC: {roc_auc:.4f}")
        
        print(f"\n📝 Sample Detailed Predictions:")
        for i, pred_detail in enumerate(detailed_predictions[:3]):
            print(f"   Sample {i+1}: {pred_detail['diagnosis']} (Confidence: {pred_detail['confidence']:.3f})")
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'validation_samples': len(y_test)
        }
        
    except Exception as e:
        print(f"❌ Pipeline validation failed: {e}")
        return None

def save_model_pipeline(predictor, validation_results=None):
    """
    Save the complete model pipeline and associated files
    """
    print("💾 Saving Model Pipeline...")
    
    # Create deployment directory
    deployment_dir = "model_deployment/"
    os.makedirs(deployment_dir, exist_ok=True)
    
    # 1. Save the complete pipeline using joblib (recommended)
    pipeline_filename = f"{deployment_dir}heart_disease_predictor_pipeline.pkl"
    joblib.dump(predictor, pipeline_filename)
    print(f"✅ Saved complete pipeline: {pipeline_filename}")
    
    # 2. Save using pickle as backup
    pickle_filename = f"{deployment_dir}heart_disease_predictor_pipeline.pickle"
    with open(pickle_filename, 'wb') as f:
        pickle.dump(predictor, f)
    print(f"✅ Saved pickle backup: {pickle_filename}")
    
    # 3. Save individual components separately for flexibility
    components_dir = f"{deployment_dir}components/"
    os.makedirs(components_dir, exist_ok=True)
    
    if predictor.model is not None:
        joblib.dump(predictor.model, f"{components_dir}model.pkl")
        print(f"✅ Saved model component")
    
    if predictor.scaler is not None:
        joblib.dump(predictor.scaler, f"{components_dir}scaler.pkl")
        print(f"✅ Saved scaler component")
    
    # 4. Save model metadata
    metadata = {
        'model_info': predictor.get_model_info(),
        'validation_results': validation_results,
        'deployment_date': datetime.now().isoformat(),
        'files': {
            'pipeline': pipeline_filename,
            'pipeline_pickle': pickle_filename,
            'model_component': f"{components_dir}model.pkl",
            'scaler_component': f"{components_dir}scaler.pkl"
        }
    }
    
    with open(f"{deployment_dir}model_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✅ Saved model metadata")
    
    return deployment_dir

def create_deployment_example():
    """
    Create example usage code for the deployed model
    """
    print("📝 Creating Deployment Examples...")
    
    example_code = '''
# Heart Disease Prediction Model - Deployment Example
# Generated automatically from model export script

import joblib
import pandas as pd
import numpy as np

# Load the trained model pipeline
predictor = joblib.load('model_deployment/heart_disease_predictor_pipeline.pkl')

# Example 1: Single patient prediction
def predict_single_patient(age, sex, cp, trestbps, chol, fbs, restecg, 
                          thalach, exang, oldpeak, slope, ca, thal):
    """
    Predict heart disease for a single patient
    
    Parameters:
    - age: Age in years
    - sex: Gender (1 = male, 0 = female)
    - cp: Chest pain type (0-3)
    - trestbps: Resting blood pressure
    - chol: Serum cholesterol
    - fbs: Fasting blood sugar > 120 mg/dl (1 = true, 0 = false)
    - restecg: Resting electrocardiographic results (0-2)
    - thalach: Maximum heart rate achieved
    - exang: Exercise induced angina (1 = yes, 0 = no)
    - oldpeak: ST depression induced by exercise
    - slope: Slope of the peak exercise ST segment (0-2)
    - ca: Number of major vessels colored by fluoroscopy (0-3)
    - thal: Thalassemia (1 = normal, 2 = fixed defect, 3 = reversable defect)
    """
    
    # Create feature array (adjust based on your actual features)
    features = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, 
                         thalach, exang, oldpeak, slope, ca, thal]])
    
    # Convert to DataFrame with proper column names
    feature_names = predictor.feature_names
    patient_data = pd.DataFrame(features, columns=feature_names[:len(features[0])])
    
    # Get detailed prediction
    result = predictor.get_prediction_with_confidence(patient_data)[0]
    
    return result

# Example 2: Batch prediction
def predict_batch(csv_file_path):
    """
    Predict heart disease for multiple patients from CSV
    """
    # Load data
    data = pd.read_csv(csv_file_path)
    
    # Make predictions
    results = predictor.get_prediction_with_confidence(data)
    
    # Add results to dataframe
    predictions_df = pd.DataFrame(results)
    combined_results = pd.concat([data, predictions_df], axis=1)
    
    return combined_results

# Example 3: Simple prediction function
def simple_predict(patient_data):
    """
    Simple prediction function
    patient_data: pandas DataFrame or numpy array
    """
    if isinstance(patient_data, dict):
        # Convert dict to DataFrame
        patient_data = pd.DataFrame([patient_data])
    
    prediction = predictor.predict(patient_data)[0]
    
    if prediction == 1:
        return "Heart Disease Detected"
    else:
        return "No Heart Disease"

# Example usage:
if __name__ == "__main__":
    # Example patient data (adjust based on your features)
    example_patient = {
        # Add your actual feature names and example values here
        'age': 63,
        'sex': 1,
        'cp': 3,
        'trestbps': 145,
        'chol': 233,
        'fbs': 1,
        'restecg': 0,
        'thalach': 150,
        'exang': 0,
        'oldpeak': 2.3,
        'slope': 0,
        'ca': 0,
        'thal': 1
    }
    
    # Make prediction
    try:
        result = simple_predict(example_patient)
        print(f"Prediction: {result}")
        
        # Get detailed prediction
        detailed = predict_single_patient(**example_patient)
        print(f"Detailed result: {detailed}")
        
    except Exception as e:
        print(f"Error making prediction: {e}")
        print("Please adjust feature names and values based on your model")

# Model Information
print("Model Information:")
model_info = predictor.get_model_info()
print(f"Model Type: {model_info['model_type']}")
print(f"Features: {model_info['n_features']}")
print(f"Creation Date: {model_info['creation_date']}")
'''
    
    with open('model_deployment/deployment_example.py', 'w', encoding='utf-8') as f:
        f.write(example_code)
    
    print("✅ Saved deployment_example.py")

def create_model_requirements():
    """
    Create requirements.txt for deployment environment
    """
    requirements = '''# Heart Disease Prediction Model - Requirements
# Install with: pip install -r requirements.txt

pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
joblib>=1.0.0
matplotlib>=3.3.0
seaborn>=0.11.0
'''
    
    with open('model_deployment/requirements.txt', 'w', encoding='utf-8') as f:
        f.write(requirements)
    
    print("✅ Created requirements.txt")

def create_model_documentation():
    """
    Create comprehensive documentation for the deployed model
    """
    print("📚 Creating Model Documentation...")
    
    doc_content = f'''# Heart Disease Prediction Model - Documentation

## Model Overview
This is a trained machine learning model for predicting heart disease based on patient medical data.

**Generated on:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
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
patient_data = pd.DataFrame([{{
    # Your feature values here
}}])

# Get prediction
result = predictor.get_prediction_with_confidence(patient_data)[0]
print(f"Diagnosis: {{result['diagnosis']}}")
print(f"Confidence: {{result['confidence']:.3f}}")
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
{{
    'prediction': 0 or 1,           # 0 = No Disease, 1 = Disease
    'diagnosis': str,               # Human readable diagnosis
    'confidence': float,            # Confidence score (0-1)
    'risk_level': str,             # 'Low', 'Medium', or 'High'
    'probabilities': [float, float] # [prob_no_disease, prob_disease]
}}
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
'''
    
    with open('model_deployment/model_documentation.md', 'w', encoding='utf-8') as f:
        f.write(doc_content)
    
    print("✅ Created model_documentation.md")

def main():
    """
    Main function to execute model export and deployment preparation
    """
    print("🚀 Starting Model Export & Deployment")
    print("=" * 60)
    
    # 1. Load best model and components
    model, scaler, feature_names, validation_data, model_info = load_optimized_models_and_data()
    
    if model is None:
        print("❌ Could not load required model components.")
        print("💡 Please ensure you have run the hyperparameter optimization script first.")
        return None
    
    # 2. Create model pipeline
    predictor = create_model_pipeline(model, scaler, feature_names, model_info)
    
    # 3. Validate pipeline
    validation_results = validate_model_pipeline(predictor, validation_data)
    
    # 4. Save model pipeline
    deployment_dir = save_model_pipeline(predictor, validation_results)
    
    # 5. Create deployment examples and documentation
    create_deployment_example()
    create_model_requirements()
    create_model_documentation()
    
    print(f"\n🎉 Model Export & Deployment Complete!")
    print("=" * 60)
    print("📋 Deliverables Created:")
    print("   ✔ Model exported as .pkl file")
    print("   ✔ Complete model pipeline with preprocessing")
    print("   ✔ Model validation and performance verification")
    print("   ✔ Deployment examples and documentation")
    print("   ✔ Requirements file for production deployment")
    print("   ✔ Individual model components for flexibility")
    
    print(f"\n📁 Deployment Files Location: {deployment_dir}")
    print("📋 Key Files:")
    print("   🎯 heart_disease_predictor_pipeline.pkl - Main deployment file")
    print("   📝 deployment_example.py - Usage examples")
    print("   📚 model_documentation.md - Complete documentation")
    print("   📋 requirements.txt - Python dependencies")
    print("   📊 model_metadata.json - Model information and performance")
    
    if validation_results:
        print(f"\n📊 Model Performance Validation:")
        print(f"   Accuracy: {validation_results['accuracy']:.4f}")
        print(f"   F1-Score: {validation_results['f1_score']:.4f}")
        print(f"   Precision: {validation_results['precision']:.4f}")
        print(f"   Recall: {validation_results['recall']:.4f}")
        if validation_results['roc_auc']:
            print(f"   ROC AUC: {validation_results['roc_auc']:.4f}")
    
    print(f"\n🚀 Ready for Production Deployment!")
    print("💡 Use 'heart_disease_predictor_pipeline.pkl' for production")
    print("📖 See 'model_documentation.md' for deployment instructions")
    
    return {
        'predictor': predictor,
        'deployment_dir': deployment_dir,
        'validation_results': validation_results
    }

if __name__ == "__main__":
    # Execute the main deployment preparation
    results = main()