
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
