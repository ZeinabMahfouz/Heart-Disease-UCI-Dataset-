import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                           classification_report, confusion_matrix, roc_curve, auc, 
                           roc_auc_score, precision_recall_curve)
import joblib
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_dataset(file_path=None):
    """
    Load dataset from various sources (feature selected, PCA, or original)
    Priority: Feature Selected > PCA > Original preprocessed data
    """
    datasets_to_try = [
        "feature_selected_top_10.csv",  # From step 2.3
        "feature_selected_top_15.csv",  # From step 2.3
        "pca_dataset_90pct.csv",        # From step 2.2
        "model_ready_data.csv"          # From step 2.1
    ]
    
    if file_path:
        datasets_to_try.insert(0, file_path)
    
    for dataset_path in datasets_to_try:
        try:
            data = pd.read_csv(dataset_path)
            print(f"✅ Data loaded successfully from: {dataset_path}")
            print(f"Dataset shape: {data.shape}")
            return data, dataset_path
        except FileNotFoundError:
            continue
    
    print("❌ No suitable dataset found!")
    print("💡 Available options:")
    print("   - feature_selected_top_X.csv (from step 2.3 - RECOMMENDED)")
    print("   - pca_dataset_Xpct.csv (from step 2.2)")
    print("   - model_ready_data.csv (from step 2.1)")
    return None, None

def prepare_classification_data(data, target_column='target'):
    """
    Prepare features and target for classification
    """
    if target_column in data.columns:
        X = data.drop(target_column, axis=1)
        y = data[target_column]
    else:
        # Assume last column is target
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]
    
    feature_names = X.columns.tolist()
    
    print(f"✅ Features: {X.shape[1]} columns")
    print(f"✅ Target distribution:")
    print(y.value_counts())
    print(f"✅ Class balance: {y.value_counts(normalize=True).round(3).to_dict()}")
    
    return X, y, feature_names

def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split data into training and testing sets with stratification
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"✅ Data split completed:")
    print(f"   Training set: {X_train.shape[0]} samples ({X_train.shape[0]/(X_train.shape[0]+X_test.shape[0])*100:.1f}%)")
    print(f"   Testing set: {X_test.shape[0]} samples ({X_test.shape[0]/(X_train.shape[0]+X_test.shape[0])*100:.1f}%)")
    print(f"   Training target distribution: {y_train.value_counts().to_dict()}")
    print(f"   Testing target distribution: {y_test.value_counts().to_dict()}")
    
    return X_train, X_test, y_train, y_test

def scale_features(X_train, X_test):
    """
    Scale features using StandardScaler (important for SVM and Logistic Regression)
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("✅ Features scaled using StandardScaler")
    
    return X_train_scaled, X_test_scaled, scaler

def initialize_models():
    """
    Initialize all classification models with default parameters
    """
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(random_state=42, probability=True)  # probability=True for ROC curves
    }
    
    print("✅ Models initialized:")
    for name in models.keys():
        print(f"   - {name}")
    
    return models

def train_models(models, X_train_scaled, X_train, y_train):
    """
    Train all models (scaled features for LR/SVM, original for tree-based)
    """
    trained_models = {}
    training_scores = {}
    
    print("\n🚀 Training Models...")
    
    for name, model in models.items():
        print(f"\n📈 Training {name}...")
        
        # Use scaled features for LR and SVM, original for tree-based models
        if name in ['Logistic Regression', 'SVM']:
            X_train_use = X_train_scaled
        else:
            X_train_use = X_train
        
        # Train model
        model.fit(X_train_use, y_train)
        
        # Cross-validation score
        cv_scores = cross_val_score(model, X_train_use, y_train, cv=5, scoring='accuracy')
        
        trained_models[name] = model
        training_scores[name] = {
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'cv_scores': cv_scores
        }
        
        print(f"✅ {name} - CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    return trained_models, training_scores

def evaluate_models(trained_models, X_test_scaled, X_test, y_test):
    """
    Evaluate all trained models on test set
    """
    evaluation_results = {}
    predictions = {}
    probabilities = {}
    
    print("\n📊 Evaluating Models on Test Set...")
    
    for name, model in trained_models.items():
        print(f"\n🔍 Evaluating {name}...")
        
        # Use appropriate features
        if name in ['Logistic Regression', 'SVM']:
            X_test_use = X_test_scaled
        else:
            X_test_use = X_test
        
        # Make predictions
        y_pred = model.predict(X_test_use)
        y_pred_proba = model.predict_proba(X_test_use)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        evaluation_results[name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }
        
        predictions[name] = y_pred
        probabilities[name] = y_pred_proba
        
        print(f"✅ {name} Results:")
        print(f"   Accuracy:  {accuracy:.4f}")
        print(f"   Precision: {precision:.4f}")
        print(f"   Recall:    {recall:.4f}")
        print(f"   F1-Score:  {f1:.4f}")
        print(f"   ROC AUC:   {roc_auc:.4f}")
    
    return evaluation_results, predictions, probabilities

def create_performance_comparison(evaluation_results, training_scores):
    """
    Create comprehensive performance comparison
    """
    print("\n📈 Creating Performance Comparison...")
    
    # Create comparison DataFrame
    comparison_data = []
    
    for model_name in evaluation_results.keys():
        comparison_data.append({
            'Model': model_name,
            'CV_Accuracy': training_scores[model_name]['cv_mean'],
            'CV_Std': training_scores[model_name]['cv_std'],
            'Test_Accuracy': evaluation_results[model_name]['accuracy'],
            'Precision': evaluation_results[model_name]['precision'],
            'Recall': evaluation_results[model_name]['recall'],
            'F1_Score': evaluation_results[model_name]['f1_score'],
            'ROC_AUC': evaluation_results[model_name]['roc_auc']
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df.sort_values('Test_Accuracy', ascending=False)
    
    print("✅ Performance Comparison Table:")
    print(comparison_df.round(4))
    
    return comparison_df

def visualize_model_performance(evaluation_results, probabilities, y_test, comparison_df):
    """
    Create comprehensive visualizations for model performance
    """
    print("\n📊 Creating Performance Visualizations...")
    
    fig = plt.figure(figsize=(20, 16))
    
    # 1. Performance Metrics Comparison (Bar Plot)
    plt.subplot(2, 3, 1)
    metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
    model_names = list(evaluation_results.keys())
    
    x = np.arange(len(metrics))
    width = 0.2
    colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFB366']
    
    for i, model in enumerate(model_names):
        values = [evaluation_results[model][metric] for metric in metrics]
        plt.bar(x + i*width, values, width, label=model, color=colors[i], alpha=0.7)
    
    plt.xlabel('Metrics')
    plt.ylabel('Score')
    plt.title('Model Performance Comparison')
    plt.xticks(x + width*1.5, [m.replace('_', ' ').title() for m in metrics])
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. ROC Curves
    plt.subplot(2, 3, 2)
    colors_roc = ['red', 'blue', 'green', 'orange']
    
    for i, (model_name, y_pred_proba) in enumerate(probabilities.items()):
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        auc_score = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=colors_roc[i], linewidth=2, 
                label=f'{model_name} (AUC = {auc_score:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves Comparison')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    # 3. Precision-Recall Curves
    plt.subplot(2, 3, 3)
    
    for i, (model_name, y_pred_proba) in enumerate(probabilities.items()):
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        plt.plot(recall, precision, color=colors_roc[i], linewidth=2, label=model_name)
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 4. Confusion Matrices
    plt.subplot(2, 3, 4)
    # Show confusion matrix for best performing model
    best_model = comparison_df.iloc[0]['Model']
    cm = evaluation_results[best_model]['confusion_matrix']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Disease', 'Disease'],
                yticklabels=['No Disease', 'Disease'])
    plt.title(f'Confusion Matrix - {best_model}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # 5. Model Accuracy Comparison
    plt.subplot(2, 3, 5)
    models = comparison_df['Model']
    test_acc = comparison_df['Test_Accuracy']
    cv_acc = comparison_df['CV_Accuracy']
    
    x_pos = np.arange(len(models))
    plt.bar(x_pos - 0.2, cv_acc, 0.4, label='CV Accuracy', alpha=0.7, color='lightblue')
    plt.bar(x_pos + 0.2, test_acc, 0.4, label='Test Accuracy', alpha=0.7, color='lightcoral')
    
    plt.xlabel('Models')
    plt.ylabel('Accuracy')
    plt.title('Cross-Validation vs Test Accuracy')
    plt.xticks(x_pos, models, rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 6. F1-Score vs ROC AUC Scatter
    plt.subplot(2, 3, 6)
    f1_scores = comparison_df['F1_Score']
    roc_aucs = comparison_df['ROC_AUC']
    
    plt.scatter(f1_scores, roc_aucs, s=100, alpha=0.7, c=colors[:len(models)])
    
    for i, model in enumerate(models):
        plt.annotate(model, (f1_scores.iloc[i], roc_aucs.iloc[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    plt.xlabel('F1-Score')
    plt.ylabel('ROC AUC')
    plt.title('F1-Score vs ROC AUC')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('classification_models_performance.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Performance visualizations saved as 'classification_models_performance.png'")

def hyperparameter_tuning(X_train_scaled, X_train, y_train, best_models=None):
    """
    Perform hyperparameter tuning for the best performing models
    """
    if best_models is None:
        best_models = ['Random Forest', 'Logistic Regression']
    
    print(f"\n🔧 Hyperparameter Tuning for: {best_models}")
    
    param_grids = {
        'Logistic Regression': {
            'C': [0.1, 1, 10, 100],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear']
        },
        'Decision Tree': {
            'max_depth': [3, 5, 7, 10, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        },
        'Random Forest': {
            'n_estimators': [100, 200, 300],
            'max_depth': [5, 10, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        },
        'SVM': {
            'C': [0.1, 1, 10, 100],
            'kernel': ['rbf', 'linear'],
            'gamma': ['scale', 'auto']
        }
    }
    
    tuned_models = {}
    
    for model_name in best_models:
        if model_name not in param_grids:
            continue
            
        print(f"\n🎯 Tuning {model_name}...")
        
        # Select appropriate model and data
        if model_name == 'Logistic Regression':
            base_model = LogisticRegression(random_state=42, max_iter=1000)
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
        
        # Grid search with cross-validation
        grid_search = GridSearchCV(
            base_model, 
            param_grids[model_name],
            cv=5,
            scoring='accuracy',
            n_jobs=-1
        )
        
        grid_search.fit(X_use, y_train)
        
        tuned_models[model_name] = {
            'model': grid_search.best_estimator_,
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_
        }
        
        print(f"✅ {model_name} Best CV Score: {grid_search.best_score_:.4f}")
        print(f"✅ {model_name} Best Parameters: {grid_search.best_params_}")
    
    return tuned_models

def generate_classification_report(evaluation_results, y_test, predictions):
    """
    Generate detailed classification reports for all models
    """
    print("\n📋 Detailed Classification Reports:")
    
    reports = {}
    
    for model_name, y_pred in predictions.items():
        print(f"\n{'='*50}")
        print(f"CLASSIFICATION REPORT - {model_name}")
        print(f"{'='*50}")
        
        report = classification_report(y_test, y_pred, target_names=['No Disease', 'Disease'])
        print(report)
        reports[model_name] = report
    
    return reports

def save_models_and_results(trained_models, scaler, evaluation_results, comparison_df, 
                          classification_reports, dataset_source):
    """
    Save trained models and all results
    """
    print("\n💾 Saving Models and Results...")
    
    # Save trained models
    models_dir = "trained_models/"
    import os
    os.makedirs(models_dir, exist_ok=True)
    
    for name, model in trained_models.items():
        model_filename = f"{models_dir}{name.lower().replace(' ', '_')}_model.pkl"
        joblib.dump(model, model_filename)
        print(f"✅ Saved {model_filename}")
    
    # Save scaler
    joblib.dump(scaler, f"{models_dir}feature_scaler.pkl")
    print(f"✅ Saved {models_dir}feature_scaler.pkl")
    
    # Save performance comparison
    comparison_df.to_csv('model_performance_comparison.csv', index=False)
    print("✅ Saved model_performance_comparison.csv")
    
    # Save detailed results
    import json
    
    # Convert numpy types to native Python types for JSON serialization
    results_summary = {}
    for model_name, metrics in evaluation_results.items():
        results_summary[model_name] = {
            'accuracy': float(metrics['accuracy']),
            'precision': float(metrics['precision']),
            'recall': float(metrics['recall']),
            'f1_score': float(metrics['f1_score']),
            'roc_auc': float(metrics['roc_auc']),
            'confusion_matrix': metrics['confusion_matrix'].tolist()
        }
    
    final_results = {
        'dataset_source': dataset_source,
        'model_performance': results_summary,
        'best_model': comparison_df.iloc[0]['Model'],
        'best_model_metrics': {
            'accuracy': float(comparison_df.iloc[0]['Test_Accuracy']),
            'f1_score': float(comparison_df.iloc[0]['F1_Score']),
            'roc_auc': float(comparison_df.iloc[0]['ROC_AUC'])
        }
    }
    
    with open('classification_results_summary.json', 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print("✅ Saved classification_results_summary.json")

def main():
    """
    Main function to execute complete classification analysis
    """
    print("🚀 Starting Supervised Learning - Classification Analysis")
    print("=" * 70)
    
    # 1. Load dataset
    data, dataset_source = load_dataset()
    if data is None:
        return
    
    # 2. Prepare data
    X, y, feature_names = prepare_classification_data(data)
    
    # 3. Split data (80% train, 20% test)
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2)
    
    # 4. Scale features
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    
    # 5. Initialize models
    models = initialize_models()
    
    # 6. Train models
    trained_models, training_scores = train_models(models, X_train_scaled, X_train, y_train)
    
    # 7. Evaluate models
    evaluation_results, predictions, probabilities = evaluate_models(
        trained_models, X_test_scaled, X_test, y_test)
    
    # 8. Create performance comparison
    comparison_df = create_performance_comparison(evaluation_results, training_scores)
    
    # 9. Generate detailed classification reports
    classification_reports = generate_classification_report(evaluation_results, y_test, predictions)
    
    # 10. Create visualizations
    visualize_model_performance(evaluation_results, probabilities, y_test, comparison_df)
    
    # 11. Hyperparameter tuning for best models
    best_model_name = comparison_df.iloc[0]['Model']
    second_best = comparison_df.iloc[1]['Model'] if len(comparison_df) > 1 else None
    best_models_to_tune = [best_model_name]
    if second_best:
        best_models_to_tune.append(second_best)
    
    tuned_models = hyperparameter_tuning(X_train_scaled, X_train, y_train, best_models_to_tune)
    
    # 12. Save all results
    save_models_and_results(trained_models, scaler, evaluation_results, 
                          comparison_df, classification_reports, dataset_source)
    
    print(f"\n🎉 Classification Analysis Complete!")
    print("=" * 70)
    print("📋 Deliverables Created:")
    print("   ✔ Trained models with performance metrics")
    print("   ✔ ROC Curves & AUC Scores visualization")
    print("   ✔ Comprehensive performance comparison")
    print("   ✔ Confusion matrices and classification reports")
    print("   ✔ Hyperparameter-tuned models")
    print("   ✔ All models saved for future use")
    
    print(f"\n🏆 Best Performing Model: {comparison_df.iloc[0]['Model']}")
    print(f"    Test Accuracy: {comparison_df.iloc[0]['Test_Accuracy']:.4f}")
    print(f"    F1-Score: {comparison_df.iloc[0]['F1_Score']:.4f}")
    print(f"    ROC AUC: {comparison_df.iloc[0]['ROC_AUC']:.4f}")
    
    return trained_models, evaluation_results, comparison_df

if __name__ == "__main__":
    # Execute the main analysis
    results = main()