# Hyperparameter Optimization
# Step 2.6: Advanced Hyperparameter Tuning for Heart Disease Classification Models

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import (GridSearchCV, RandomizedSearchCV, train_test_split, 
                                   cross_val_score, StratifiedKFold)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                           classification_report, confusion_matrix, roc_curve, auc, 
                           roc_auc_score, make_scorer)
from scipy.stats import randint, uniform
import joblib
import time
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_dataset_and_baseline(file_path=None):
    """
    Load dataset and baseline results from previous steps
    """
    datasets_to_try = [
        "feature_selected_top_10.csv",  # From step 2.3 - BEST for optimization
        "feature_selected_top_15.csv",  # From step 2.3 - Alternative
        "pca_dataset_90pct.csv",        # From step 2.2
        "model_ready_data.csv"          # From step 2.1
    ]
    
    if file_path:
        datasets_to_try.insert(0, file_path)
    
    # Load dataset
    data, dataset_source = None, None
    for dataset_path in datasets_to_try:
        try:
            data = pd.read_csv(dataset_path)
            dataset_source = dataset_path
            print(f"✅ Data loaded successfully from: {dataset_path}")
            print(f"Dataset shape: {data.shape}")
            break
        except FileNotFoundError:
            continue
    
    if data is None:
        print("❌ No suitable dataset found!")
        return None, None, None
    
    # Try to load baseline results
    baseline_results = None
    try:
        import json
        with open('classification_results_summary.json', 'r') as f:
            baseline_results = json.load(f)
        print("✅ Baseline results loaded from previous classification step")
    except FileNotFoundError:
        print("⚠️  No baseline results found - will create new baseline")
    
    return data, dataset_source, baseline_results

def prepare_data_for_optimization(data, target_column='target', test_size=0.2, random_state=42):
    """
    Prepare data for hyperparameter optimization
    """
    if target_column in data.columns:
        X = data.drop(target_column, axis=1)
        y = data[target_column]
    else:
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]
    
    feature_names = X.columns.tolist()
    
    print(f"✅ Features for optimization: {X.shape[1]} columns")
    print(f"✅ Target distribution: {y.value_counts().to_dict()}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"✅ Data split: {X_train.shape[0]} train, {X_test.shape[0]} test samples")
    
    return X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_names

def define_hyperparameter_grids():
    """
    Define comprehensive hyperparameter grids for all models
    """
    print("🎯 Defining Hyperparameter Grids...")
    
    # Define parameter grids for GridSearchCV (smaller, focused grids)
    grid_params = {
        'Logistic Regression': {
            'C': [0.01, 0.1, 1, 10, 100],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear'],
            'max_iter': [1000]
        },
        
        'Decision Tree': {
            'max_depth': [3, 5, 7, 10, None],
            'min_samples_split': [2, 5, 10, 20],
            'min_samples_leaf': [1, 2, 4, 8],
            'criterion': ['gini', 'entropy']
        },
        
        'Random Forest': {
            'n_estimators': [100, 200, 300],
            'max_depth': [5, 10, 15, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2']
        },
        
        'SVM': {
            'C': [0.1, 1, 10, 100],
            'kernel': ['rbf', 'linear'],
            'gamma': ['scale', 'auto', 0.001, 0.01]
        },
        
        'Gradient Boosting': {
            'n_estimators': [100, 200],
            'learning_rate': [0.05, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'min_samples_split': [2, 5, 10]
        },
        
        'KNN': {
            'n_neighbors': [3, 5, 7, 9, 11, 15],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan']
        }
    }
    
    # Define parameter distributions for RandomizedSearchCV (larger, broader distributions)
    random_params = {
        'Logistic Regression': {
            'C': uniform(0.01, 100),
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear'],
            'max_iter': [1000, 2000]
        },
        
        'Decision Tree': {
            'max_depth': randint(3, 20),
            'min_samples_split': randint(2, 20),
            'min_samples_leaf': randint(1, 10),
            'criterion': ['gini', 'entropy'],
            'max_features': [None, 'sqrt', 'log2']
        },
        
        'Random Forest': {
            'n_estimators': randint(100, 500),
            'max_depth': [3, 5, 7, 10, 15, 20, None],
            'min_samples_split': randint(2, 20),
            'min_samples_leaf': randint(1, 10),
            'max_features': ['sqrt', 'log2', None],
            'bootstrap': [True, False]
        },
        
        'SVM': {
            'C': uniform(0.1, 100),
            'kernel': ['rbf', 'linear', 'poly'],
            'gamma': ['scale', 'auto'] + list(uniform(0.001, 0.1).rvs(5))
        },
        
        'Gradient Boosting': {
            'n_estimators': randint(100, 500),
            'learning_rate': uniform(0.01, 0.3),
            'max_depth': randint(3, 10),
            'min_samples_split': randint(2, 20),
            'min_samples_leaf': randint(1, 10),
            'subsample': uniform(0.8, 0.2)
        },
        
        'KNN': {
            'n_neighbors': randint(3, 20),
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan', 'minkowski'],
            'leaf_size': randint(10, 50)
        }
    }
    
    print("✅ Hyperparameter grids defined for 6 algorithms")
    print("✅ GridSearchCV: Focused parameter grids")
    print("✅ RandomizedSearchCV: Broad parameter distributions")
    
    return grid_params, random_params

def create_baseline_models():
    """
    Create baseline models with default parameters
    """
    baseline_models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(random_state=42, probability=True),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'KNN': KNeighborsClassifier(),
        'Naive Bayes': GaussianNB()
    }
    
    return baseline_models

def evaluate_baseline_models(models, X_train, X_train_scaled, y_train, X_test, X_test_scaled, y_test):
    """
    Evaluate baseline models to establish performance benchmarks
    """
    print("\n📊 Evaluating Baseline Models...")
    
    baseline_results = {}
    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for name, model in models.items():
        print(f"\n🔄 Training {name}...")
        
        # Use scaled features for algorithms that benefit from scaling
        if name in ['Logistic Regression', 'SVM', 'KNN', 'Naive Bayes']:
            X_train_use = X_train_scaled
            X_test_use = X_test_scaled
        else:
            X_train_use = X_train
            X_test_use = X_test
        
        start_time = time.time()
        
        # Train model
        model.fit(X_train_use, y_train)
        
        # Cross-validation score
        cv_scores = cross_val_score(model, X_train_use, y_train, cv=cv_strategy, scoring='accuracy')
        
        # Test predictions
        y_pred = model.predict(X_test_use)
        y_pred_proba = model.predict_proba(X_test_use)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else None
        
        training_time = time.time() - start_time
        
        baseline_results[name] = {
            'model': model,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'test_accuracy': accuracy,
            'test_precision': precision,
            'test_recall': recall,
            'test_f1': f1,
            'test_roc_auc': roc_auc,
            'training_time': training_time,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
        
        print(f"✅ {name}:")
        print(f"   CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        print(f"   Test Accuracy: {accuracy:.4f}")
        print(f"   F1-Score: {f1:.4f}")
        print(f"   Training Time: {training_time:.2f}s")
    
    return baseline_results

def perform_grid_search_optimization(grid_params, X_train, X_train_scaled, y_train):
    """
    Perform GridSearchCV optimization for all models
    """
    print("\n🎯 Performing GridSearchCV Optimization...")
    
    grid_results = {}
    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for model_name, params in grid_params.items():
        print(f"\n🔍 GridSearchCV for {model_name}...")
        start_time = time.time()
        
        # Create base model
        if model_name == 'Logistic Regression':
            base_model = LogisticRegression(random_state=42)
            X_use = X_train_scaled
        elif model_name == 'Decision Tree':
            base_model = DecisionTreeClassifier(random_state=42)
            X_use = X_train
        elif model_name == 'Random Forest':
            base_model = RandomForestClassifier(random_state=42)
            X_use = X_train
        elif model_name == 'SVM':
            base_model = SVC(random_state=42, probability=True)
            X_use = X_train_scaled
        elif model_name == 'Gradient Boosting':
            base_model = GradientBoostingClassifier(random_state=42)
            X_use = X_train
        elif model_name == 'KNN':
            base_model = KNeighborsClassifier()
            X_use = X_train_scaled
        
        # Perform grid search
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=params,
            cv=cv_strategy,
            scoring='accuracy',
            n_jobs=-1,
            verbose=0
        )
        
        grid_search.fit(X_use, y_train)
        
        optimization_time = time.time() - start_time
        
        grid_results[model_name] = {
            'best_model': grid_search.best_estimator_,
            'best_params': grid_search.best_params_,
            'best_cv_score': grid_search.best_score_,
            'optimization_time': optimization_time,
            'n_combinations': len(grid_search.cv_results_['params'])
        }
        
        print(f"✅ {model_name} GridSearch completed:")
        print(f"   Best CV Score: {grid_search.best_score_:.4f}")
        print(f"   Best Parameters: {grid_search.best_params_}")
        print(f"   Combinations tested: {len(grid_search.cv_results_['params'])}")
        print(f"   Optimization time: {optimization_time:.2f}s")
    
    return grid_results

def perform_random_search_optimization(random_params, X_train, X_train_scaled, y_train, n_iter=100):
    """
    Perform RandomizedSearchCV optimization for all models
    """
    print(f"\n🎲 Performing RandomizedSearchCV Optimization (n_iter={n_iter})...")
    
    random_results = {}
    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for model_name, params in random_params.items():
        print(f"\n🔍 RandomizedSearchCV for {model_name}...")
        start_time = time.time()
        
        # Create base model
        if model_name == 'Logistic Regression':
            base_model = LogisticRegression(random_state=42)
            X_use = X_train_scaled
        elif model_name == 'Decision Tree':
            base_model = DecisionTreeClassifier(random_state=42)
            X_use = X_train
        elif model_name == 'Random Forest':
            base_model = RandomForestClassifier(random_state=42)
            X_use = X_train
        elif model_name == 'SVM':
            base_model = SVC(random_state=42, probability=True)
            X_use = X_train_scaled
        elif model_name == 'Gradient Boosting':
            base_model = GradientBoostingClassifier(random_state=42)
            X_use = X_train
        elif model_name == 'KNN':
            base_model = KNeighborsClassifier()
            X_use = X_train_scaled
        
        # Perform randomized search
        random_search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=params,
            n_iter=n_iter,
            cv=cv_strategy,
            scoring='accuracy',
            n_jobs=-1,
            random_state=42,
            verbose=0
        )
        
        random_search.fit(X_use, y_train)
        
        optimization_time = time.time() - start_time
        
        random_results[model_name] = {
            'best_model': random_search.best_estimator_,
            'best_params': random_search.best_params_,
            'best_cv_score': random_search.best_score_,
            'optimization_time': optimization_time,
            'n_combinations': n_iter
        }
        
        print(f"✅ {model_name} RandomizedSearch completed:")
        print(f"   Best CV Score: {random_search.best_score_:.4f}")
        print(f"   Best Parameters: {random_search.best_params_}")
        print(f"   Combinations tested: {n_iter}")
        print(f"   Optimization time: {optimization_time:.2f}s")
    
    return random_results

def evaluate_optimized_models(grid_results, random_results, X_test, X_test_scaled, y_test):
    """
    Evaluate optimized models on test set
    """
    print("\n📊 Evaluating Optimized Models on Test Set...")
    
    optimized_results = {}
    
    # Evaluate GridSearchCV results
    print("\n🎯 GridSearchCV Results:")
    for model_name, result in grid_results.items():
        model = result['best_model']
        
        # Use appropriate test data
        if model_name in ['Logistic Regression', 'SVM', 'KNN']:
            X_test_use = X_test_scaled
        else:
            X_test_use = X_test
        
        # Make predictions
        y_pred = model.predict(X_test_use)
        y_pred_proba = model.predict_proba(X_test_use)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else None
        
        optimized_results[f"{model_name}_Grid"] = {
            'model': model,
            'method': 'GridSearchCV',
            'best_params': result['best_params'],
            'cv_score': result['best_cv_score'],
            'test_accuracy': accuracy,
            'test_precision': precision,
            'test_recall': recall,
            'test_f1': f1,
            'test_roc_auc': roc_auc,
            'optimization_time': result['optimization_time'],
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
        
        print(f"✅ {model_name} (Grid):")
        print(f"   Test Accuracy: {accuracy:.4f}")
        print(f"   F1-Score: {f1:.4f}")
        print(f"   ROC AUC: {roc_auc:.4f}" if roc_auc else "   ROC AUC: N/A")
    
    # Evaluate RandomizedSearchCV results
    print("\n🎲 RandomizedSearchCV Results:")
    for model_name, result in random_results.items():
        model = result['best_model']
        
        # Use appropriate test data
        if model_name in ['Logistic Regression', 'SVM', 'KNN']:
            X_test_use = X_test_scaled
        else:
            X_test_use = X_test
        
        # Make predictions
        y_pred = model.predict(X_test_use)
        y_pred_proba = model.predict_proba(X_test_use)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else None
        
        optimized_results[f"{model_name}_Random"] = {
            'model': model,
            'method': 'RandomizedSearchCV',
            'best_params': result['best_params'],
            'cv_score': result['best_cv_score'],
            'test_accuracy': accuracy,
            'test_precision': precision,
            'test_recall': recall,
            'test_f1': f1,
            'test_roc_auc': roc_auc,
            'optimization_time': result['optimization_time'],
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
        
        print(f"✅ {model_name} (Random):")
        print(f"   Test Accuracy: {accuracy:.4f}")
        print(f"   F1-Score: {f1:.4f}")
        print(f"   ROC AUC: {roc_auc:.4f}" if roc_auc else "   ROC AUC: N/A")
    
    return optimized_results

def compare_baseline_vs_optimized(baseline_results, optimized_results):
    """
    Create comprehensive comparison between baseline and optimized models
    """
    print("\n🔍 Comparing Baseline vs Optimized Performance...")
    
    comparison_data = []
    
    # Process baseline results
    for model_name, result in baseline_results.items():
        if model_name != 'Naive Bayes':  # Skip Naive Bayes as it's not optimized
            comparison_data.append({
                'Model': model_name,
                'Method': 'Baseline',
                'CV_Score': result['cv_mean'],
                'Test_Accuracy': result['test_accuracy'],
                'Test_Precision': result['test_precision'],
                'Test_Recall': result['test_recall'],
                'Test_F1': result['test_f1'],
                'Test_ROC_AUC': result['test_roc_auc'] if result['test_roc_auc'] else 0,
                'Training_Time': result['training_time']
            })
    
    # Process optimized results
    for model_key, result in optimized_results.items():
        model_name = model_key.replace('_Grid', '').replace('_Random', '')
        method = result['method']
        
        comparison_data.append({
            'Model': model_name,
            'Method': method,
            'CV_Score': result['cv_score'],
            'Test_Accuracy': result['test_accuracy'],
            'Test_Precision': result['test_precision'],
            'Test_Recall': result['test_recall'],
            'Test_F1': result['test_f1'],
            'Test_ROC_AUC': result['test_roc_auc'] if result['test_roc_auc'] else 0,
            'Training_Time': result['optimization_time']
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Calculate improvements
    improvement_data = []
    
    for model_name in comparison_df['Model'].unique():
        model_data = comparison_df[comparison_df['Model'] == model_name]
        
        baseline_data = model_data[model_data['Method'] == 'Baseline']
        grid_data = model_data[model_data['Method'] == 'GridSearchCV']
        random_data = model_data[model_data['Method'] == 'RandomizedSearchCV']
        
        if len(baseline_data) > 0:
            baseline_f1 = baseline_data['Test_F1'].iloc[0]
            baseline_acc = baseline_data['Test_Accuracy'].iloc[0]
            
            if len(grid_data) > 0:
                grid_f1 = grid_data['Test_F1'].iloc[0]
                grid_acc = grid_data['Test_Accuracy'].iloc[0]
                f1_improvement_grid = ((grid_f1 - baseline_f1) / baseline_f1) * 100
                acc_improvement_grid = ((grid_acc - baseline_acc) / baseline_acc) * 100
                
                improvement_data.append({
                    'Model': model_name,
                    'Method': 'GridSearchCV',
                    'F1_Improvement_%': f1_improvement_grid,
                    'Accuracy_Improvement_%': acc_improvement_grid
                })
            
            if len(random_data) > 0:
                random_f1 = random_data['Test_F1'].iloc[0]
                random_acc = random_data['Test_Accuracy'].iloc[0]
                f1_improvement_random = ((random_f1 - baseline_f1) / baseline_f1) * 100
                acc_improvement_random = ((random_acc - baseline_acc) / baseline_acc) * 100
                
                improvement_data.append({
                    'Model': model_name,
                    'Method': 'RandomizedSearchCV',
                    'F1_Improvement_%': f1_improvement_random,
                    'Accuracy_Improvement_%': acc_improvement_random
                })
    
    improvement_df = pd.DataFrame(improvement_data)
    
    print("✅ Performance comparison completed")
    print(f"✅ Total models compared: {len(comparison_df['Model'].unique())}")
    
    return comparison_df, improvement_df

def create_optimization_visualizations(comparison_df, improvement_df, optimized_results, y_test):
    """
    Create comprehensive visualizations for hyperparameter optimization results
    """
    print("\n📊 Creating Optimization Visualizations...")
    
    fig = plt.figure(figsize=(24, 20))
    
    # 1. Performance Comparison - Test Accuracy
    plt.subplot(3, 4, 1)
    pivot_acc = comparison_df.pivot(index='Model', columns='Method', values='Test_Accuracy')
    pivot_acc.plot(kind='bar', ax=plt.gca(), alpha=0.7)
    plt.title('Test Accuracy: Baseline vs Optimized')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=45)
    plt.legend(title='Method')
    plt.grid(True, alpha=0.3)
    
    # 2. Performance Comparison - F1 Score
    plt.subplot(3, 4, 2)
    pivot_f1 = comparison_df.pivot(index='Model', columns='Method', values='Test_F1')
    pivot_f1.plot(kind='bar', ax=plt.gca(), alpha=0.7)
    plt.title('F1-Score: Baseline vs Optimized')
    plt.ylabel('F1-Score')
    plt.xticks(rotation=45)
    plt.legend(title='Method')
    plt.grid(True, alpha=0.3)
    
    # 3. ROC AUC Comparison
    plt.subplot(3, 4, 3)
    pivot_roc = comparison_df.pivot(index='Model', columns='Method', values='Test_ROC_AUC')
    pivot_roc.plot(kind='bar', ax=plt.gca(), alpha=0.7)
    plt.title('ROC AUC: Baseline vs Optimized')
    plt.ylabel('ROC AUC')
    plt.xticks(rotation=45)
    plt.legend(title='Method')
    plt.grid(True, alpha=0.3)
    
    # 4. Training Time Comparison
    plt.subplot(3, 4, 4)
    pivot_time = comparison_df.pivot(index='Model', columns='Method', values='Training_Time')
    pivot_time.plot(kind='bar', ax=plt.gca(), alpha=0.7, logy=True)
    plt.title('Training Time: Baseline vs Optimized (Log Scale)')
    plt.ylabel('Time (seconds)')
    plt.xticks(rotation=45)
    plt.legend(title='Method')
    plt.grid(True, alpha=0.3)
    
    # 5. Improvement Percentages - F1 Score
    plt.subplot(3, 4, 5)
    if not improvement_df.empty:
        pivot_imp_f1 = improvement_df.pivot(index='Model', columns='Method', values='F1_Improvement_%')
        pivot_imp_f1.plot(kind='bar', ax=plt.gca(), alpha=0.7)
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        plt.title('F1-Score Improvement (%)')
        plt.ylabel('Improvement (%)')
        plt.xticks(rotation=45)
        plt.legend(title='Method')
        plt.grid(True, alpha=0.3)
    
    # 6. Improvement Percentages - Accuracy
    plt.subplot(3, 4, 6)
    if not improvement_df.empty:
        pivot_imp_acc = improvement_df.pivot(index='Model', columns='Method', values='Accuracy_Improvement_%')
        pivot_imp_acc.plot(kind='bar', ax=plt.gca(), alpha=0.7)
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        plt.title('Accuracy Improvement (%)')
        plt.ylabel('Improvement (%)')
        plt.xticks(rotation=45)
        plt.legend(title='Method')
        plt.grid(True, alpha=0.3)
    
    # 7. ROC Curves for Best Models
    plt.subplot(3, 4, 7)
    
    # Get best performing models from each optimization method
    best_models = []
    for method in ['GridSearchCV', 'RandomizedSearchCV']:
        method_data = comparison_df[comparison_df['Method'] == method]
        if not method_data.empty:
            best_model_row = method_data.loc[method_data['Test_F1'].idxmax()]
            # Fix key naming issue
            if method == 'GridSearchCV':
                key_suffix = 'Grid'
            else:  # RandomizedSearchCV
                key_suffix = 'Random'
            
            model_key = f"{best_model_row['Model']}_{key_suffix}"
            
            if model_key in optimized_results:
                best_models.append({
                    'name': f"{best_model_row['Model']} ({method[:4]})",
                    'probabilities': optimized_results[model_key]["probabilities"]
                })
            else:
                print(f"⚠️  Warning: {model_key} not found in optimized_results")
    
    colors = ['red', 'blue', 'green', 'orange']
    for i, model_data in enumerate(best_models):
        if model_data['probabilities'] is not None:
            fpr, tpr, _ = roc_curve(y_test, model_data['probabilities'])
            auc_score = auc(fpr, tpr)
            plt.plot(fpr, tpr, color=colors[i], linewidth=2, 
                    label=f"{model_data['name']} (AUC = {auc_score:.3f})")
    
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves - Best Optimized Models')
    plt.legend(loc="lower right", fontsize=8)
    plt.grid(True, alpha=0.3)
    
    # 8. Cross-Validation Score vs Test Score
    plt.subplot(3, 4, 8)
    cv_scores = comparison_df['CV_Score']
    test_scores = comparison_df['Test_Accuracy']
    methods = comparison_df['Method']
    
    colors_map = {'Baseline': 'red', 'GridSearchCV': 'blue', 'RandomizedSearchCV': 'green'}
    for method in methods.unique():
        mask = methods == method
        plt.scatter(cv_scores[mask], test_scores[mask], 
                   c=colors_map.get(method, 'black'), label=method, alpha=0.7, s=60)
    
    # Add diagonal line for perfect correlation
    min_val = min(cv_scores.min(), test_scores.min())
    max_val = max(cv_scores.max(), test_scores.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)
    
    plt.xlabel('Cross-Validation Score')
    plt.ylabel('Test Accuracy')
    plt.title('CV Score vs Test Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 9. Model Performance Heatmap
    plt.subplot(3, 4, 9)
    heatmap_data = comparison_df.pivot_table(
        index='Model', 
        columns='Method', 
        values='Test_F1', 
        aggfunc='mean'
    )
    sns.heatmap(heatmap_data, annot=True, cmap='YlOrRd', fmt='.3f', cbar_kws={'label': 'F1-Score'})
    plt.title('F1-Score Performance Heatmap')
    plt.ylabel('Model')
    plt.xlabel('Method')
    
    # 10. Precision vs Recall Scatter
    plt.subplot(3, 4, 10)
    for method in comparison_df['Method'].unique():
        method_data = comparison_df[comparison_df['Method'] == method]
        plt.scatter(method_data['Test_Recall'], method_data['Test_Precision'], 
                   label=method, alpha=0.7, s=60)
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision vs Recall')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 11. Best Model Comparison (Top 3)
    plt.subplot(3, 4, 11)
    top_models = comparison_df.nlargest(6, 'Test_F1')  # Top 6 to show variety
    
    metrics = ['Test_Accuracy', 'Test_Precision', 'Test_Recall', 'Test_F1']
    x = np.arange(len(metrics))
    width = 0.12
    
    for i, (_, model_row) in enumerate(top_models.head(5).iterrows()):  # Top 5 models
        values = [model_row[metric] for metric in metrics]
        plt.bar(x + i*width, values, width, 
                label=f"{model_row['Model']} ({model_row['Method'][:4]})", alpha=0.7)
    
    plt.xlabel('Metrics')
    plt.ylabel('Score')
    plt.title('Top 5 Models - Detailed Metrics')
    plt.xticks(x + width*2, [m.replace('Test_', '') for m in metrics])
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    plt.grid(True, alpha=0.3)
    
    # 12. Optimization Time Analysis
    plt.subplot(3, 4, 12)
    optimization_data = comparison_df[comparison_df['Method'] != 'Baseline']
    
    # Group by model and method
    time_comparison = optimization_data.pivot(
        index='Model', 
        columns='Method', 
        values='Training_Time'
    )
    
    time_comparison.plot(kind='bar', ax=plt.gca(), alpha=0.7, logy=True)
    plt.title('Optimization Time Comparison (Log Scale)')
    plt.ylabel('Time (seconds)')
    plt.xticks(rotation=45)
    plt.legend(title='Method')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('hyperparameter_optimization_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Optimization visualizations saved as 'hyperparameter_optimization_results.png'")

def find_best_overall_model(comparison_df, optimized_results):
    """
    Identify the best overall model across all optimization methods
    """
    print("\n🏆 Finding Best Overall Model...")
    
    # Find best model based on F1-score
    best_row = comparison_df.loc[comparison_df['Test_F1'].idxmax()]
    best_model_name = best_row['Model']
    best_method = best_row['Method']
    
    if best_method == 'Baseline':
        best_model_key = best_model_name
        best_model = None  # Will need to retrain
        best_params = None
    else:
        # Fix key naming consistency
        if best_method == 'GridSearchCV':
            key_suffix = 'Grid'
        else:  # RandomizedSearchCV
            key_suffix = 'Random'
        
        best_model_key = f"{best_model_name}_{key_suffix}"
        
        if best_model_key in optimized_results:
            best_model = optimized_results[best_model_key]['model']
            best_params = optimized_results[best_model_key]['best_params']
        else:
            print(f"⚠️  Warning: {best_model_key} not found in optimized_results")
            best_model = None
            best_params = None
    
    print(f"🎯 Best Overall Model: {best_model_name}")
    print(f"🔧 Optimization Method: {best_method}")
    print(f"📊 Performance Metrics:")
    print(f"   Test Accuracy: {best_row['Test_Accuracy']:.4f}")
    print(f"   Test Precision: {best_row['Test_Precision']:.4f}")
    print(f"   Test Recall: {best_row['Test_Recall']:.4f}")
    print(f"   Test F1-Score: {best_row['Test_F1']:.4f}")
    print(f"   Test ROC AUC: {best_row['Test_ROC_AUC']:.4f}")
    
    if best_method != 'Baseline' and best_params is not None:
        print(f"🎛️  Best Parameters: {best_params}")
    
    return {
        'model_name': best_model_name,
        'method': best_method,
        'model': best_model,
        'metrics': {
            'accuracy': best_row['Test_Accuracy'],
            'precision': best_row['Test_Precision'],
            'recall': best_row['Test_Recall'],
            'f1_score': best_row['Test_F1'],
            'roc_auc': best_row['Test_ROC_AUC']
        },
        'params': best_params
    }

def save_optimization_results(comparison_df, improvement_df, optimized_results, best_model_info, 
                            dataset_source, scaler):
    """
    Save all hyperparameter optimization results
    """
    print("\n💾 Saving Optimization Results...")
    
    # Create directory for optimized models
    import os
    models_dir = "optimized_models/"
    os.makedirs(models_dir, exist_ok=True)
    
    # Save all optimized models
    for model_key, result in optimized_results.items():
        model_filename = f"{models_dir}{model_key.lower().replace(' ', '_')}_optimized.pkl"
        joblib.dump(result['model'], model_filename)
        print(f"✅ Saved {model_filename}")
    
    # Save best overall model separately
    if best_model_info['model'] is not None:
        best_model_filename = f"{models_dir}best_model_{best_model_info['model_name'].lower().replace(' ', '_')}.pkl"
        joblib.dump(best_model_info['model'], best_model_filename)
        print(f"🏆 Saved best model: {best_model_filename}")
    
    # Save scaler
    joblib.dump(scaler, f"{models_dir}feature_scaler_optimized.pkl")
    print(f"✅ Saved feature scaler")
    
    # Save comparison results
    comparison_df.to_csv('optimization_performance_comparison.csv', index=False)
    print("✅ Saved optimization_performance_comparison.csv")
    
    if not improvement_df.empty:
        improvement_df.to_csv('optimization_improvement_analysis.csv', index=False)
        print("✅ Saved optimization_improvement_analysis.csv")
    
    # Save comprehensive results summary
    import json
    
    # Convert numpy types for JSON serialization
    def convert_numpy_types(obj):
        if hasattr(obj, 'item'):
            return obj.item()
        elif hasattr(obj, 'tolist'):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        else:
            return obj
    
    # Prepare optimization summary
    optimization_summary = {
        'dataset_source': dataset_source,
        'best_overall_model': {
            'name': best_model_info['model_name'],
            'optimization_method': best_model_info['method'],
            'parameters': best_model_info['params'],
            'performance_metrics': {k: convert_numpy_types(v) for k, v in best_model_info['metrics'].items()}
        },
        'optimization_methods_compared': ['GridSearchCV', 'RandomizedSearchCV'],
        'models_optimized': list(set([key.split('_')[0] + '_' + key.split('_')[1] if len(key.split('_')) > 2 
                                    else key.replace('_Grid', '').replace('_Random', '') 
                                    for key in optimized_results.keys()])),
        'performance_summary': {
            'best_accuracy': float(comparison_df['Test_Accuracy'].max()),
            'best_f1_score': float(comparison_df['Test_F1'].max()),
            'best_roc_auc': float(comparison_df['Test_ROC_AUC'].max()),
            'average_improvement': {
                'f1_score': float(improvement_df['F1_Improvement_%'].mean()) if not improvement_df.empty else 0,
                'accuracy': float(improvement_df['Accuracy_Improvement_%'].mean()) if not improvement_df.empty else 0
            }
        }
    }
    
    with open('hyperparameter_optimization_summary.json', 'w') as f:
        json.dump(optimization_summary, f, indent=2, default=convert_numpy_types)
    
    print("✅ Saved hyperparameter_optimization_summary.json")
    
    # Save detailed optimization parameters for reproducibility
    optimization_params = {}
    for model_key, result in optimized_results.items():
        optimization_params[model_key] = {
            'best_parameters': result['best_params'],
            'cv_score': convert_numpy_types(result['cv_score']),
            'optimization_method': result['method'],
            'optimization_time': convert_numpy_types(result['optimization_time'])
        }
    
    with open('optimization_parameters_detailed.json', 'w') as f:
        json.dump(optimization_params, f, indent=2, default=convert_numpy_types)
    
    print("✅ Saved optimization_parameters_detailed.json")

def generate_optimization_report(comparison_df, improvement_df, best_model_info, optimization_time_total):
    """
    Generate a comprehensive optimization report
    """
    print("\n📋 Generating Optimization Report...")
    
    report = f"""
# HYPERPARAMETER OPTIMIZATION REPORT
{'='*50}

## EXECUTIVE SUMMARY
- Total models optimized: {len(comparison_df['Model'].unique())}
- Optimization methods used: GridSearchCV, RandomizedSearchCV
- Total optimization time: {optimization_time_total:.2f} seconds
- Best overall model: {best_model_info['model_name']} ({best_model_info['method']})

## BEST MODEL PERFORMANCE
- Model: {best_model_info['model_name']}
- Optimization Method: {best_model_info['method']}
- Test Accuracy: {best_model_info['metrics']['accuracy']:.4f}
- Test F1-Score: {best_model_info['metrics']['f1_score']:.4f}
- Test ROC AUC: {best_model_info['metrics']['roc_auc']:.4f}

## OPTIMIZATION RESULTS SUMMARY
"""
    
    if not improvement_df.empty:
        avg_f1_improvement = improvement_df['F1_Improvement_%'].mean()
        avg_acc_improvement = improvement_df['Accuracy_Improvement_%'].mean()
        best_f1_improvement = improvement_df['F1_Improvement_%'].max()
        best_acc_improvement = improvement_df['Accuracy_Improvement_%'].max()
        
        report += f"""
### IMPROVEMENT ANALYSIS
- Average F1-Score Improvement: {avg_f1_improvement:.2f}%
- Average Accuracy Improvement: {avg_acc_improvement:.2f}%
- Best F1-Score Improvement: {best_f1_improvement:.2f}%
- Best Accuracy Improvement: {best_acc_improvement:.2f}%
"""
    
    report += f"""
### TOP 3 PERFORMING MODELS
"""
    
    top_3 = comparison_df.nlargest(3, 'Test_F1')
    for i, (_, row) in enumerate(top_3.iterrows(), 1):
        report += f"""
{i}. {row['Model']} ({row['Method']})
   - F1-Score: {row['Test_F1']:.4f}
   - Accuracy: {row['Test_Accuracy']:.4f}
   - ROC AUC: {row['Test_ROC_AUC']:.4f}
"""
    
    report += f"""
### METHOD COMPARISON
- GridSearchCV: Exhaustive search, guaranteed to find optimal within grid
- RandomizedSearchCV: Efficient sampling, explores broader parameter space

### OPTIMIZATION IMPACT
"""
    
    baseline_models = comparison_df[comparison_df['Method'] == 'Baseline']
    optimized_models = comparison_df[comparison_df['Method'] != 'Baseline']
    
    if len(baseline_models) > 0 and len(optimized_models) > 0:
        baseline_avg_f1 = baseline_models['Test_F1'].mean()
        optimized_avg_f1 = optimized_models['Test_F1'].mean()
        overall_improvement = ((optimized_avg_f1 - baseline_avg_f1) / baseline_avg_f1) * 100
        
        report += f"""
- Baseline average F1-Score: {baseline_avg_f1:.4f}
- Optimized average F1-Score: {optimized_avg_f1:.4f}
- Overall improvement: {overall_improvement:.2f}%
"""
    
    report += f"""
## RECOMMENDATIONS
1. Use {best_model_info['model_name']} as the primary model
2. Optimization method: {best_model_info['method']}
3. Key hyperparameters: {best_model_info['params'] if best_model_info['params'] else 'Default parameters'}

## FILES GENERATED
- optimized_models/ - Directory with all optimized models
- hyperparameter_optimization_results.png - Comprehensive visualizations
- optimization_performance_comparison.csv - Detailed performance comparison
- hyperparameter_optimization_summary.json - Machine-readable summary
"""
    
    # Save report
    with open('hyperparameter_optimization_report.txt', 'w') as f:
        f.write(report)
    
    print("✅ Saved hyperparameter_optimization_report.txt")
    print("\n📊 Report Summary:")
    print(f"🏆 Best Model: {best_model_info['model_name']} ({best_model_info['method']})")
    print(f"📈 Best F1-Score: {best_model_info['metrics']['f1_score']:.4f}")
    print(f"⏱️  Total Optimization Time: {optimization_time_total:.2f} seconds")

def main():
    """
    Main function to execute comprehensive hyperparameter optimization
    """
    print("🚀 Starting Hyperparameter Optimization Analysis")
    print("=" * 70)
    
    start_time = time.time()
    
    # 1. Load dataset and baseline results
    data, dataset_source, baseline_results = load_dataset_and_baseline()
    if data is None:
        return
    
    # 2. Prepare data
    X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_names = \
        prepare_data_for_optimization(data)
    
    # 3. Define hyperparameter grids
    grid_params, random_params = define_hyperparameter_grids()
    
    # 4. Create and evaluate baseline models (if not available)
    baseline_models = create_baseline_models()
    baseline_results = evaluate_baseline_models(
        baseline_models, X_train, X_train_scaled, y_train, X_test, X_test_scaled, y_test
    )
    
    # 5. Perform GridSearchCV optimization
    grid_results = perform_grid_search_optimization(grid_params, X_train, X_train_scaled, y_train)
    
    # 6. Perform RandomizedSearchCV optimization
    random_results = perform_random_search_optimization(
        random_params, X_train, X_train_scaled, y_train, n_iter=100
    )
    
    # 7. Evaluate optimized models
    optimized_results = evaluate_optimized_models(
        grid_results, random_results, X_test, X_test_scaled, y_test
    )
    
    # 8. Compare baseline vs optimized performance
    comparison_df, improvement_df = compare_baseline_vs_optimized(baseline_results, optimized_results)
    
    # 9. Create comprehensive visualizations
    create_optimization_visualizations(comparison_df, improvement_df, optimized_results, y_test)
    
    # 10. Find best overall model
    best_model_info = find_best_overall_model(comparison_df, optimized_results)
    
    # 11. Save all results
    save_optimization_results(
        comparison_df, improvement_df, optimized_results, best_model_info, dataset_source, scaler
    )
    
    # 12. Generate comprehensive report
    total_time = time.time() - start_time
    generate_optimization_report(comparison_df, improvement_df, best_model_info, total_time)
    
    print(f"\n🎉 Hyperparameter Optimization Complete!")
    print("=" * 70)
    print("📋 Deliverables Created:")
    print("   ✔ Best performing model with optimized hyperparameters")
    print("   ✔ Comprehensive comparison: GridSearchCV vs RandomizedSearchCV")
    print("   ✔ Baseline vs Optimized performance analysis")
    print("   ✔ All optimized models saved for deployment")
    print("   ✔ Detailed optimization report and visualizations")
    print("   ✔ Performance improvement analysis")
    
    print(f"\n🏆 FINAL RESULTS:")
    print(f"Best Model: {best_model_info['model_name']}")
    print(f"Optimization Method: {best_model_info['method']}")
    print(f"Test F1-Score: {best_model_info['metrics']['f1_score']:.4f}")
    print(f"Test Accuracy: {best_model_info['metrics']['accuracy']:.4f}")
    print(f"Test ROC AUC: {best_model_info['metrics']['roc_auc']:.4f}")
    
    if not improvement_df.empty:
        avg_improvement = improvement_df['F1_Improvement_%'].mean()
        print(f"Average F1-Score Improvement: {avg_improvement:.2f}%")
    
    return {
        'best_model': best_model_info,
        'comparison_results': comparison_df,
        'optimization_results': optimized_results,
        'improvement_analysis': improvement_df
    }

if __name__ == "__main__":
    # Execute the main analysis
    results = main()