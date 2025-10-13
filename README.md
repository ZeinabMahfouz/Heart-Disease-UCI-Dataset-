# ❤️ Heart Disease Prediction System

<div align="center">
  
![Python Version](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.0+-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Live%20Demo-brightgreen.svg)

<h3>🚀 Complete End-to-End Machine Learning Pipeline for Heart Disease Prediction</h3>
<p>A comprehensive ML system featuring data preprocessing, advanced modeling, hyperparameter optimization, and a live web interface with public deployment.</p>

</div>

## 📋 Table of Contents
- [Project Overview](#project-overview)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Model Performance](#model-performance)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Pipeline Components](#pipeline-components)
- [Web Application](#web-application)
- [Deployment](#deployment)
- [Technologies Used](#technologies-used)
- [Contributing](#contributing)
- [License](#license)
- [Author](#author)

## 🎯 Project Overview

This project implements a complete end-to-end machine learning pipeline for heart disease prediction using the UCI Heart Disease dataset.

**Dataset Source**: [UCI Heart Disease Dataset](https://archive.ics.uci.edu/dataset/45/heart+disease)

**Objective**: Develop a production-ready ML system that can accurately predict heart disease risk from patient medical data with real-time web interface.

## ✨ Key Features

### 🧠 Machine Learning
- Multi-Algorithm Comparison (Random Forest, SVM, Logistic Regression, etc.)
- Hyperparameter Optimization with GridSearchCV & RandomizedSearchCV
- Feature Engineering with PCA and statistical feature selection
- Unsupervised clustering analysis

### 📊 Data Science Pipeline
- Comprehensive EDA and statistical analysis
- Advanced data preprocessing and validation
- Multiple feature selection methods
- Cross-validation and performance metrics

### 🌐 Web Application
- Streamlit-based professional interface
- Real-time predictions with confidence scoring
- Interactive visualizations
- Medical recommendations

## 🏗️ System Architecture
Raw Data → Data Preprocessing → Feature Engineering → Model Training
↓ ↓ ↓
EDA Analysis PCA & Feature Multiple Algorithms
Selection & Optimization
↓ ↓ ↓
Visualizations Best Features Model Evaluation
↓ ↓ ↓
Model Export → Web App → Public Deployment

text

## 📊 Model Performance

### Best Model Results
| Model | Accuracy | Precision | Recall | F1-Score | ROC AUC |
|-------|----------|-----------|--------|----------|---------|
| **Random Forest (Optimized)** | **0.918** | **0.923** | **0.900** | **0.911** | **0.954** |
| Logistic Regression | 0.852 | 0.857 | 0.840 | 0.848 | 0.901 |
| SVM (RBF) | 0.885 | 0.889 | 0.875 | 0.882 | 0.923 |

### Clinical Metrics
- **Sensitivity (Recall)**: 90.0% - Correctly identifies 9/10 heart disease cases
- **Specificity**: 93.6% - Correctly identifies healthy patients
- **Positive Predictive Value**: 92.3%

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/ZeinabMahfouz/Heart-Disease-UCI-Dataset-.git
cd heart-disease-prediction
2. Install Dependencies
bash
pip install -r requirements.txt
3. Run the Pipeline
bash
# Data preprocessing
python data_preprocessing.py

# Feature engineering
python pca_analysis.py
python feature_selection.py

# Model training
python classification_models.py
python hyperparameter_optimization.py

# Launch web app
streamlit run heart_disease_streamlit_app.py
4. Access Application
Local: http://localhost:8501

Public: Run python ngrok_deployment.py

💻 Installation
Requirements
Python 3.8+

8GB RAM (recommended)

Setup
bash
# Create virtual environment
python -m venv heart_disease_env
# On Windows: heart_disease_env\Scripts\activate
# On Mac/Linux: source heart_disease_env/bin/activate

# Install dependencies
pip install -r requirements.txt
📁 Project Structure
text
heart-disease-prediction/
│
├── data/                              # Dataset files
│   ├── heart_disease_dataset.csv
│   ├── model_ready_data.csv
│   └── feature_selected_top_10.csv
│
├── clustering_models/                 # ML models
│   ├── trained_models/
│   ├── optimized_models/
│   └── model_deployment/
│
├── results/                           # Analysis results
│   ├── feature_importance_analysis.png
│   ├── classification_models_performance.png
│   └── hyperparameter_optimization_results.png
│
├── scripts/                           # Main pipeline
│   ├── data_preprocessing.py
│   ├── pca_analysis.py
│   ├── feature_selection.py
│   ├── classification_models.py
│   ├── clustering_analysis.py
│   ├── hyperparameter_optimization.py
│   ├── model_export_deployment.py
│   ├── heart_disease_streamlit_app.py
│   └── ngrok_deployment.py
│
├── requirements.txt
├── streamlit_requirements.txt
└── README.md
🔬 Pipeline Components
1. Data Preprocessing (data_preprocessing.py)
Missing value handling and data cleaning

Feature encoding and scaling

Exploratory Data Analysis

2. Dimensionality Reduction (pca_analysis.py)
Principal Component Analysis

Variance explanation (85%, 90%, 95% retention)

PCA visualizations

3. Feature Selection (feature_selection.py)
Statistical methods (Chi-square, F-test, mutual information)

Model-based importance (Random Forest, XGBoost)

Recursive Feature Elimination

4. Model Training (classification_models.py)
Multiple algorithms comparison

Cross-validation and performance metrics

ROC curves and confusion matrices

5. Clustering Analysis (clustering_analysis.py)
K-Means and Hierarchical clustering

Cluster validation and analysis

Pattern discovery

6. Hyperparameter Optimization (hyperparameter_optimization.py)
Grid Search and Randomized Search

Performance tracking

Optimization analysis

7. Web Application (heart_disease_streamlit_app.py)
Interactive user interface

Real-time predictions

Medical recommendations

8. Deployment (ngrok_deployment.py)
Public access via ngrok

Secure HTTPS tunneling

🌐 Web Application Features
Patient Data Input Form: Comprehensive medical parameters

Real-time Risk Assessment: Instant predictions with confidence scores

Interactive Visualizations: Data exploration tools

Medical Recommendations: Personalized health advice

Responsive Design: Works on desktop and mobile

🚀 Deployment
Local Deployment
bash
streamlit run heart_disease_streamlit_app.py
Public Deployment with Ngrok
bash
python ngrok_deployment.py
Cloud Options
Streamlit Cloud

Heroku

Docker containerization

🛠️ Technologies Used
Machine Learning
Python, Scikit-learn, Pandas, NumPy

Matplotlib, Seaborn, Plotly

Joblib, SciPy

Web & Deployment
Streamlit, Ngrok

HTML/CSS, JavaScript

Development
Git, Jupyter Notebooks

VS Code, Docker

🤝 Contributing
We welcome contributions! Here's how you can help:

Report Bugs: Open an issue with detailed description

Suggest Features: Share your ideas for improvements

Code Contributions: Submit pull requests

Improve Documentation: Help enhance docs and tutorials

Contribution Process
Fork the repository

Create a feature branch: git checkout -b feature/your-feature

Make your changes and test

Submit a pull request

📜 License
This project is licensed under the MIT License - see the LICENSE file for details.

👨‍💻 Author
Zeinab Mahfouz

GitHub: @ZeinabMahfouz

Project Repository: Heart Disease Prediction

🙏 Acknowledgments
Data Source: UCI Machine Learning Repository - Heart Disease Dataset

Tools: Streamlit, Scikit-learn, and the open-source community

References: Medical research on heart disease prediction

<div align="center"><h3>⭐ If you find this project helpful, please give it a star! ⭐</h3>
Built with ❤️ for advancing healthcare through AI

</div> ```
