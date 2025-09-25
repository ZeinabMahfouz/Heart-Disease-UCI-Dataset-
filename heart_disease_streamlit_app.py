import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import json
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configure Streamlit page
st.set_page_config(
    page_title="Heart Disease Prediction System",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #e74c3c;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2c3e50;
        margin-bottom: 1rem;
    }
    .prediction-box {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
        font-weight: bold;
        font-size: 1.2rem;
    }
    .high-risk {
        background-color: #ffebee;
        border: 2px solid #f44336;
        color: #d32f2f;
    }
    .low-risk {
        background-color: #e8f5e8;
        border: 2px solid #4caf50;
        color: #388e3c;
    }
    .medium-risk {
        background-color: #fff3e0;
        border: 2px solid #ff9800;
        color: #f57c00;
    }
    .info-box {
        background-color: #f0f8ff;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #2196f3;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

class SimpleHeartDiseasePredictor:
    """
    Simple Heart Disease Prediction class that works with any sklearn model
    """
    
    def __init__(self, model=None, scaler=None):
        self.model = model
        self.scaler = scaler
        self.feature_names = [
            'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 
            'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
        ]
    
    def predict(self, X):
        """Make predictions on new data"""
        if self.model is None:
            # Return dummy prediction if no model
            return np.array([0])
        
        # Apply scaling if scaler exists
        if self.scaler is not None:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X
        
        predictions = self.model.predict(X_scaled)
        return predictions
    
    def predict_proba(self, X):
        """Get prediction probabilities"""
        if self.model is None:
            # Return dummy probabilities if no model
            return np.array([[0.7, 0.3]])
        
        # Apply scaling if scaler exists
        if self.scaler is not None:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X
        
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(X_scaled)
        else:
            # For models without predict_proba, create dummy probabilities
            predictions = self.model.predict(X_scaled)
            probabilities = np.array([[0.8, 0.2] if pred == 0 else [0.3, 0.7] for pred in predictions])
        
        return probabilities
    
    def get_prediction_with_confidence(self, X):
        """Get predictions with confidence scores and interpretation"""
        predictions = self.predict(X)
        probabilities = self.predict_proba(X)
        confidence_scores = np.max(probabilities, axis=1)
        
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
                'probabilities': probabilities[i].tolist()
            }
            results.append(result)
        
        return results

@st.cache_resource
def load_model():
    """Load any available trained model"""
    model_files_to_try = [
        'model_deployment/heart_disease_predictor_pipeline.pkl',
        'optimized_models/random_forest_grid_optimized.pkl',
        'optimized_models/logistic_regression_grid_optimized.pkl',
        'trained_models/random_forest_model.pkl',
        'trained_models/logistic_regression_model.pkl'
    ]
    
    scaler_files_to_try = [
        'model_deployment/components/scaler.pkl',
        'optimized_models/feature_scaler_optimized.pkl',
        'trained_models/feature_scaler.pkl'
    ]
    
    # Try to load model
    model = None
    model_source = None
    for model_file in model_files_to_try:
        try:
            if model_file.endswith('pipeline.pkl'):
                # This is the complete pipeline
                predictor = joblib.load(model_file)
                st.success(f"✅ Complete pipeline loaded from {model_file}")
                return predictor
            else:
                # This is just the model
                model = joblib.load(model_file)
                model_source = model_file
                st.success(f"✅ Model loaded from {model_file}")
                break
        except FileNotFoundError:
            continue
        except Exception as e:
            st.warning(f"⚠️ Error loading {model_file}: {e}")
            continue
    
    # Try to load scaler
    scaler = None
    for scaler_file in scaler_files_to_try:
        try:
            scaler = joblib.load(scaler_file)
            st.success(f"✅ Scaler loaded from {scaler_file}")
            break
        except FileNotFoundError:
            continue
        except Exception as e:
            continue
    
    if model is not None:
        # Create predictor with model and scaler
        predictor = SimpleHeartDiseasePredictor(model=model, scaler=scaler)
        return predictor
    else:
        st.error("❌ No model files found. The app will run in demo mode.")
        st.info("To use real predictions, run one of these scripts first:")
        st.code("python hyperparameter_optimization.py")
        st.code("python classification_models.py")
        st.code("python model_export_deployment.py")
        
        # Return demo predictor
        return SimpleHeartDiseasePredictor()

@st.cache_data
def load_sample_data():
    """Load sample data for visualization"""
    datasets_to_try = [
        "feature_selected_top_10.csv",
        "feature_selected_top_15.csv", 
        "model_ready_data.csv"
    ]
    
    for dataset_path in datasets_to_try:
        try:
            data = pd.read_csv(dataset_path)
            st.success(f"✅ Sample data loaded from {dataset_path}")
            return data
        except FileNotFoundError:
            continue
    
    # Create synthetic sample data for demo
    st.info("📊 Using synthetic data for demonstration")
    np.random.seed(42)
    sample_data = pd.DataFrame({
        'age': np.random.normal(54, 10, 1000).astype(int),
        'sex': np.random.choice([0, 1], 1000),
        'cp': np.random.choice([0, 1, 2, 3], 1000),
        'trestbps': np.random.normal(130, 20, 1000).astype(int),
        'chol': np.random.normal(240, 50, 1000).astype(int),
        'fbs': np.random.choice([0, 1], 1000),
        'restecg': np.random.choice([0, 1, 2], 1000),
        'thalach': np.random.normal(150, 25, 1000).astype(int),
        'exang': np.random.choice([0, 1], 1000),
        'oldpeak': np.random.exponential(1, 1000).round(1),
        'slope': np.random.choice([0, 1, 2], 1000),
        'ca': np.random.choice([0, 1, 2, 3], 1000),
        'thal': np.random.choice([1, 2, 3], 1000),
        'target': np.random.choice([0, 1], 1000)
    })
    return sample_data

def get_feature_descriptions():
    """Get descriptions for all features"""
    descriptions = {
        'age': 'Age in years',
        'sex': 'Gender (1 = Male, 0 = Female)',
        'cp': 'Chest Pain Type (0: Typical Angina, 1: Atypical Angina, 2: Non-Anginal Pain, 3: Asymptomatic)',
        'trestbps': 'Resting Blood Pressure (mm Hg)',
        'chol': 'Serum Cholesterol (mg/dl)',
        'fbs': 'Fasting Blood Sugar > 120 mg/dl (1 = True, 0 = False)',
        'restecg': 'Resting ECG (0: Normal, 1: ST-T Wave Abnormality, 2: Left Ventricular Hypertrophy)',
        'thalach': 'Maximum Heart Rate Achieved',
        'exang': 'Exercise Induced Angina (1 = Yes, 0 = No)',
        'oldpeak': 'ST Depression Induced by Exercise',
        'slope': 'Slope of Peak Exercise ST Segment (0: Upsloping, 1: Flat, 2: Downsloping)',
        'ca': 'Number of Major Vessels (0-3) Colored by Fluoroscopy',
        'thal': 'Thalassemia (1: Normal, 2: Fixed Defect, 3: Reversible Defect)'
    }
    return descriptions

def create_input_form(predictor):
    """Create the input form for patient data"""
    st.markdown('<div class="sub-header">Patient Information Input</div>', unsafe_allow_html=True)
    
    feature_descriptions = get_feature_descriptions()
    
    # Create input form
    with st.form("patient_data_form"):
        col1, col2, col3 = st.columns(3)
        
        patient_data = {}
        
        with col1:
            st.markdown("**Basic Information**")
            patient_data['age'] = st.number_input(
                "Age", min_value=1, max_value=120, value=50,
                help=feature_descriptions.get('age', '')
            )
            
            patient_data['sex'] = st.selectbox(
                "Gender", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male",
                help=feature_descriptions.get('sex', '')
            )
            
            cp_options = {0: "Typical Angina", 1: "Atypical Angina", 2: "Non-Anginal Pain", 3: "Asymptomatic"}
            patient_data['cp'] = st.selectbox(
                "Chest Pain Type", options=list(cp_options.keys()), 
                format_func=lambda x: cp_options[x],
                help=feature_descriptions.get('cp', '')
            )
            
            patient_data['fbs'] = st.selectbox(
                "Fasting Blood Sugar > 120 mg/dl", options=[0, 1],
                format_func=lambda x: "No" if x == 0 else "Yes",
                help=feature_descriptions.get('fbs', '')
            )
            
            patient_data['exang'] = st.selectbox(
                "Exercise Induced Angina", options=[0, 1],
                format_func=lambda x: "No" if x == 0 else "Yes",
                help=feature_descriptions.get('exang', '')
            )
        
        with col2:
            st.markdown("**Vital Signs**")
            patient_data['trestbps'] = st.number_input(
                "Resting Blood Pressure (mm Hg)", min_value=50, max_value=300, value=120,
                help=feature_descriptions.get('trestbps', '')
            )
            
            patient_data['chol'] = st.number_input(
                "Serum Cholesterol (mg/dl)", min_value=100, max_value=600, value=200,
                help=feature_descriptions.get('chol', '')
            )
            
            patient_data['thalach'] = st.number_input(
                "Maximum Heart Rate", min_value=50, max_value=250, value=150,
                help=feature_descriptions.get('thalach', '')
            )
            
            patient_data['oldpeak'] = st.number_input(
                "ST Depression", min_value=0.0, max_value=10.0, value=1.0, step=0.1,
                help=feature_descriptions.get('oldpeak', '')
            )
        
        with col3:
            st.markdown("**Clinical Tests**")
            restecg_options = {0: "Normal", 1: "ST-T Wave Abnormality", 2: "Left Ventricular Hypertrophy"}
            patient_data['restecg'] = st.selectbox(
                "Resting ECG", options=list(restecg_options.keys()),
                format_func=lambda x: restecg_options[x],
                help=feature_descriptions.get('restecg', '')
            )
            
            slope_options = {0: "Upsloping", 1: "Flat", 2: "Downsloping"}
            patient_data['slope'] = st.selectbox(
                "ST Segment Slope", options=list(slope_options.keys()),
                format_func=lambda x: slope_options[x],
                help=feature_descriptions.get('slope', '')
            )
            
            patient_data['ca'] = st.selectbox(
                "Major Vessels (Fluoroscopy)", options=[0, 1, 2, 3],
                help=feature_descriptions.get('ca', '')
            )
            
            thal_options = {1: "Normal", 2: "Fixed Defect", 3: "Reversible Defect"}
            patient_data['thal'] = st.selectbox(
                "Thalassemia", options=list(thal_options.keys()),
                format_func=lambda x: thal_options[x],
                help=feature_descriptions.get('thal', '')
            )
        
        # Submit button
        submitted = st.form_submit_button("🔬 Analyze Heart Disease Risk", type="primary")
        
        if submitted:
            return patient_data
    
    return None

def display_prediction_results(result):
    """Display prediction results with styling"""
    prediction = result['prediction']
    diagnosis = result['diagnosis']
    confidence = result['confidence']
    risk_level = result['risk_level']
    probabilities = result.get('probabilities', [0.5, 0.5])
    
    # Main prediction display
    if prediction == 1:
        st.markdown(f'''
        <div class="prediction-box high-risk">
            🚨 {diagnosis}
            <br>
            Confidence: {confidence:.1%}
            <br>
            Risk Level: {risk_level}
        </div>
        ''', unsafe_allow_html=True)
    else:
        st.markdown(f'''
        <div class="prediction-box low-risk">
            ✅ {diagnosis}
            <br>
            Confidence: {confidence:.1%}
            <br>
            Risk Level: {risk_level}
        </div>
        ''', unsafe_allow_html=True)
    
    # Detailed probability breakdown
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="No Heart Disease Probability",
            value=f"{probabilities[0]:.1%}",
            delta=None
        )
    
    with col2:
        st.metric(
            label="Heart Disease Probability", 
            value=f"{probabilities[1]:.1%}",
            delta=None
        )
    
    with col3:
        st.metric(
            label="Confidence Score",
            value=f"{confidence:.1%}",
            delta=None
        )
    
    # Probability visualization
    fig_prob = go.Figure(data=[
        go.Bar(
            x=['No Heart Disease', 'Heart Disease'],
            y=[probabilities[0], probabilities[1]],
            marker_color=['#4CAF50', '#F44336'],
            text=[f'{probabilities[0]:.1%}', f'{probabilities[1]:.1%}'],
            textposition='auto',
        )
    ])
    
    fig_prob.update_layout(
        title="Prediction Probabilities",
        yaxis_title="Probability",
        xaxis_title="Diagnosis",
        showlegend=False,
        height=400
    )
    
    st.plotly_chart(fig_prob, use_container_width=True)

def create_risk_interpretation(result, patient_data):
    """Create risk interpretation and recommendations"""
    st.markdown('<div class="sub-header">Risk Analysis & Recommendations</div>', unsafe_allow_html=True)
    
    prediction = result['prediction']
    confidence = result['confidence']
    
    # Risk interpretation
    if prediction == 1:
        if confidence > 0.8:
            interpretation = "⚠️ **High Risk**: The model indicates a strong likelihood of heart disease. Immediate medical consultation is strongly recommended."
        elif confidence > 0.6:
            interpretation = "🟡 **Moderate-High Risk**: There are concerning indicators. Please consult with a healthcare professional soon."
        else:
            interpretation = "🟠 **Uncertain High Risk**: Some risk factors are present, but the prediction is not highly confident. Medical evaluation recommended."
    else:
        if confidence > 0.8:
            interpretation = "✅ **Low Risk**: The model indicates a low likelihood of heart disease based on the provided information."
        elif confidence > 0.6:
            interpretation = "🟢 **Moderate-Low Risk**: Generally favorable indicators, but continued health monitoring is recommended."
        else:
            interpretation = "🟡 **Uncertain Low Risk**: Mixed indicators present. Consider regular health check-ups."
    
    st.markdown(f'''
    <div class="info-box">
        <h4>Risk Interpretation</h4>
        {interpretation}
    </div>
    ''', unsafe_allow_html=True)
    
    # General recommendations
    st.markdown("**General Recommendations:**")
    recommendations = [
        "🏃‍♂️ Maintain regular physical activity (consult doctor first if high risk)",
        "🥗 Follow a heart-healthy diet (low saturated fat, high fiber)",
        "🚭 Avoid smoking and limit alcohol consumption", 
        "💊 Take medications as prescribed by healthcare provider",
        "📊 Monitor blood pressure and cholesterol regularly",
        "😌 Manage stress through relaxation techniques",
        "⚖️ Maintain a healthy weight",
        "🩺 Schedule regular check-ups with healthcare provider"
    ]
    
    for rec in recommendations:
        st.markdown(f"• {rec}")
    
    # Disclaimer
    st.markdown('''
    <div class="info-box">
        <h4>⚠️ Important Disclaimer</h4>
        This prediction is based on a machine learning model and should NOT replace professional medical advice. 
        Always consult with qualified healthcare professionals for proper diagnosis and treatment decisions.
    </div>
    ''', unsafe_allow_html=True)

def create_basic_visualizations(sample_data):
    """Create basic data visualization dashboard"""
    st.markdown('<div class="sub-header">Heart Disease Data Insights</div>', unsafe_allow_html=True)
    
    if sample_data is None or sample_data.empty:
        st.warning("No data available for visualization")
        return
    
    # Basic statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_patients = len(sample_data)
        st.metric("Total Patients", total_patients)
    
    with col2:
        if 'target' in sample_data.columns:
            disease_rate = (sample_data['target'].sum() / len(sample_data)) * 100
            st.metric("Heart Disease Rate", f"{disease_rate:.1f}%")
        else:
            st.metric("Heart Disease Rate", "N/A")
    
    with col3:
        if 'age' in sample_data.columns:
            avg_age = sample_data['age'].mean()
            st.metric("Average Age", f"{avg_age:.1f} years")
        else:
            st.metric("Average Age", "N/A")
    
    with col4:
        if 'sex' in sample_data.columns:
            male_rate = (sample_data['sex'].sum() / len(sample_data)) * 100
            st.metric("Male Patients", f"{male_rate:.1f}%")
        else:
            st.metric("Male Patients", "N/A")
    
    # Simple visualizations
    if 'age' in sample_data.columns and 'target' in sample_data.columns:
        # Age distribution
        fig1 = px.histogram(
            sample_data, 
            x='age', 
            color='target',
            title='Age Distribution by Heart Disease Status',
            labels={'target': 'Heart Disease', 'age': 'Age (years)'},
            nbins=20
        )
        st.plotly_chart(fig1, use_container_width=True)
        
        # Correlation with target
        numeric_cols = sample_data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1 and 'target' in sample_data.columns:
            correlations = sample_data[numeric_cols].corr()['target'].drop('target').sort_values(key=abs, ascending=False)
            
            fig2 = go.Figure(data=[
                go.Bar(
                    x=correlations.values,
                    y=correlations.index,
                    orientation='h',
                    marker_color=['red' if x < 0 else 'green' for x in correlations.values]
                )
            ])
            fig2.update_layout(
                title="Feature Correlation with Heart Disease",
                xaxis_title="Correlation Coefficient",
                yaxis_title="Features",
                height=500
            )
            st.plotly_chart(fig2, use_container_width=True)

def main():
    """Main Streamlit application"""
    # Header
    st.markdown('<div class="main-header">❤️ Heart Disease Prediction System</div>', unsafe_allow_html=True)
    
    # Load model
    predictor = load_model()
    
    # Load sample data for visualization
    sample_data = load_sample_data()
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to",
        ["🔬 Prediction Tool", "📊 Data Insights", "ℹ️ About"]
    )
    
    if page == "🔬 Prediction Tool":
        st.markdown("Enter patient information below to get a heart disease risk assessment.")
        
        # Input form
        patient_data = create_input_form(predictor)
        
        if patient_data:
            try:
                # Convert to DataFrame
                patient_df = pd.DataFrame([patient_data])
                
                # Make prediction
                result = predictor.get_prediction_with_confidence(patient_df)[0]
                
                # Display results
                st.markdown("---")
                st.markdown('<div class="sub-header">Prediction Results</div>', unsafe_allow_html=True)
                
                display_prediction_results(result)
                
                # Risk interpretation
                create_risk_interpretation(result, patient_data)
                
            except Exception as e:
                st.error(f"Error making prediction: {e}")
                st.info("Please check that all required fields are filled correctly.")
    
    elif page == "📊 Data Insights":
        create_basic_visualizations(sample_data)
    
    else:  # About page
        st.markdown('<div class="sub-header">About This Application</div>', unsafe_allow_html=True)
        
        st.markdown("""
        ### Heart Disease Prediction System
        
        This application uses machine learning to assess heart disease risk based on patient medical data.
        
        **Features:**
        - 🔬 **Real-time Prediction**: Get instant risk assessment
        - 📊 **Data Visualization**: Explore heart disease trends and patterns
        - 🎯 **Risk Analysis**: Detailed interpretation and recommendations
        - 💡 **Educational**: Learn about heart disease risk factors
        
        ### How to Use
        
        1. **Navigate to Prediction Tool** using the sidebar
        2. **Fill in patient information** in the form
        3. **Click "Analyze Heart Disease Risk"** to get predictions
        4. **Review the results** and recommendations
        5. **Explore Data Insights** to understand heart disease patterns
        
        ### Important Notes
        
        - This tool is for **educational and research purposes only**
        - **NOT a substitute for professional medical advice**
        - Always consult healthcare professionals for medical decisions
        - Model predictions are based on historical data and may not be 100% accurate
        
        ### Getting Better Predictions
        
        For improved accuracy, run these scripts to train optimized models:
        
        ```bash
        # Train and optimize models
        python hyperparameter_optimization.py
        
        # Export trained models
        python model_export_deployment.py
        ```
        
        ### Contact & Support
        
        For technical issues or questions about this application, please refer to the project documentation.
        """)
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666; padding: 20px;'>
            <p>Heart Disease Prediction System | Built with Streamlit</p>
            <p><small>⚠️ For educational purposes only - Not for medical diagnosis</small></p>
        </div>
        """, 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()