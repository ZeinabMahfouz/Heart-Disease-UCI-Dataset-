import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, SelectKBest, chi2, f_classif, mutual_info_classif
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import classification_report, accuracy_score
import xgboost as xgb
from scipy.stats import chi2_contingency
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_preprocessed_data(file_path):
    """
    Load the preprocessed data from step 2.1
    """
    try:
        if file_path.endswith('.csv'):
            data = pd.read_csv(file_path)
        elif file_path.endswith('.pkl'):
            data = pd.read_pickle(file_path)
        elif file_path.endswith('.xlsx'):
            data = pd.read_excel(file_path)
        else:
            data = pd.read_csv(file_path)
        
        print(f"✅ Data loaded successfully!")
        print(f"Dataset shape: {data.shape}")
        print(f"Features: {list(data.columns[:-1])}")
        return data
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        print("💡 Make sure 'model_ready_data.csv' exists from step 2.1")
        return None

def prepare_data_for_selection(data, target_column='target'):
    """
    Prepare features and target for feature selection
    """
    if target_column in data.columns:
        X = data.drop(target_column, axis=1)
        y = data[target_column]
    else:
        # Assume last column is target
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]
    
    feature_names = X.columns.tolist()
    
    print(f"✅ Features prepared: {X.shape[1]} features")
    print(f"✅ Target distribution:")
    print(y.value_counts())
    
    return X, y, feature_names

def random_forest_feature_importance(X, y, feature_names, n_estimators=100):
    """
    Calculate feature importance using Random Forest
    """
    print("\n🌲 Random Forest Feature Importance Analysis...")
    
    # Split data for training
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Train Random Forest
    rf = RandomForestClassifier(n_estimators=n_estimators, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    
    # Get feature importance
    importance_scores = rf.feature_importances_
    
    # Create importance DataFrame
    rf_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': importance_scores,
        'method': 'Random Forest'
    }).sort_values('importance', ascending=False)
    
    # Model performance
    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    cv_scores = cross_val_score(rf, X, y, cv=5)
    
    print(f"✅ Random Forest Accuracy: {accuracy:.4f}")
    print(f"✅ Cross-validation Score: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"✅ Top 5 features: {rf_importance.head()['feature'].tolist()}")
    
    return rf_importance, rf

def xgboost_feature_importance(X, y, feature_names):
    """
    Calculate feature importance using XGBoost
    """
    print("\n🚀 XGBoost Feature Importance Analysis...")
    
    # Split data for training
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Train XGBoost
    xgb_model = xgb.XGBClassifier(random_state=42, eval_metric='logloss')
    xgb_model.fit(X_train, y_train)
    
    # Get feature importance
    importance_scores = xgb_model.feature_importances_
    
    # Create importance DataFrame
    xgb_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': importance_scores,
        'method': 'XGBoost'
    }).sort_values('importance', ascending=False)
    
    # Model performance
    y_pred = xgb_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    cv_scores = cross_val_score(xgb_model, X, y, cv=5)
    
    print(f"✅ XGBoost Accuracy: {accuracy:.4f}")
    print(f"✅ Cross-validation Score: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"✅ Top 5 features: {xgb_importance.head()['feature'].tolist()}")
    
    return xgb_importance, xgb_model

def recursive_feature_elimination(X, y, feature_names, n_features_to_select=10):
    """
    Apply Recursive Feature Elimination
    """
    print(f"\n🔄 Recursive Feature Elimination (selecting {n_features_to_select} features)...")
    
    # Use Random Forest as base estimator
    estimator = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # Apply RFE
    rfe = RFE(estimator=estimator, n_features_to_select=n_features_to_select)
    rfe.fit(X, y)
    
    # Get selected features
    selected_features = [feature_names[i] for i in range(len(feature_names)) if rfe.support_[i]]
    feature_rankings = rfe.ranking_
    
    # Create RFE results DataFrame
    rfe_results = pd.DataFrame({
        'feature': feature_names,
        'selected': rfe.support_,
        'ranking': feature_rankings
    }).sort_values('ranking')
    
    print(f"✅ Selected features: {selected_features}")
    
    # Evaluate performance with selected features
    X_selected = rfe.transform(X)
    cv_scores = cross_val_score(estimator, X_selected, y, cv=5)
    print(f"✅ RFE Model CV Score: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    return rfe_results, selected_features, rfe

def chi_square_feature_selection(X, y, feature_names, k=10):
    """
    Apply Chi-Square test for feature selection
    """
    print(f"\n📊 Chi-Square Feature Selection (top {k} features)...")
    
    # Ensure all features are non-negative for chi-square test
    # Apply MinMaxScaler to make all values positive
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply Chi-Square test
    chi2_selector = SelectKBest(chi2, k=k)
    chi2_selector.fit(X_scaled, y)
    
    # Get chi-square scores
    chi2_scores = chi2_selector.scores_
    chi2_p_values = chi2_selector.pvalues_
    
    # Create chi-square results DataFrame
    chi2_results = pd.DataFrame({
        'feature': feature_names,
        'chi2_score': chi2_scores,
        'p_value': chi2_p_values,
        'selected': chi2_selector.get_support()
    }).sort_values('chi2_score', ascending=False)
    
    selected_features = chi2_results[chi2_results['selected']]['feature'].tolist()
    
    print(f"✅ Selected features: {selected_features}")
    print(f"✅ Average chi-square score: {chi2_scores.mean():.4f}")
    
    return chi2_results, selected_features, chi2_selector

def mutual_information_selection(X, y, feature_names, k=10):
    """
    Apply Mutual Information for feature selection
    """
    print(f"\n🧠 Mutual Information Feature Selection (top {k} features)...")
    
    # Calculate mutual information scores
    mi_scores = mutual_info_classif(X, y, random_state=42)
    
    # Create mutual information DataFrame
    mi_results = pd.DataFrame({
        'feature': feature_names,
        'mi_score': mi_scores,
        'method': 'Mutual Information'
    }).sort_values('mi_score', ascending=False)
    
    # Select top k features
    selected_features = mi_results.head(k)['feature'].tolist()
    
    print(f"✅ Selected features: {selected_features}")
    print(f"✅ Average MI score: {mi_scores.mean():.4f}")
    
    return mi_results, selected_features

def univariate_statistical_tests(X, y, feature_names):
    """
    Apply univariate statistical tests (F-test)
    """
    print("\n📈 Univariate Statistical Tests (F-test)...")
    
    # Apply F-test
    f_scores, f_p_values = f_classif(X, y)
    
    # Create F-test results DataFrame
    f_test_results = pd.DataFrame({
        'feature': feature_names,
        'f_score': f_scores,
        'p_value': f_p_values,
        'method': 'F-test'
    }).sort_values('f_score', ascending=False)
    
    # Select significant features (p < 0.05)
    significant_features = f_test_results[f_test_results['p_value'] < 0.05]['feature'].tolist()
    
    print(f"✅ Significant features (p < 0.05): {len(significant_features)}")
    print(f"✅ Features: {significant_features}")
    
    return f_test_results, significant_features

def combine_feature_rankings(rf_importance, xgb_importance, mi_results, f_test_results, chi2_results):
    """
    Combine all feature importance methods into a single ranking
    """
    print("\n🔗 Combining Feature Rankings...")
    
    # Create a comprehensive ranking system
    all_features = rf_importance['feature'].tolist()
    
    combined_rankings = []
    
    for feature in all_features:
        # Get rankings from each method
        rf_rank = rf_importance[rf_importance['feature'] == feature].index[0] + 1
        xgb_rank = xgb_importance[xgb_importance['feature'] == feature].index[0] + 1
        mi_rank = mi_results[mi_results['feature'] == feature].index[0] + 1
        f_rank = f_test_results[f_test_results['feature'] == feature].index[0] + 1
        chi2_rank = chi2_results[chi2_results['feature'] == feature].index[0] + 1
        
        # Get actual scores
        rf_score = rf_importance[rf_importance['feature'] == feature]['importance'].iloc[0]
        xgb_score = xgb_importance[xgb_importance['feature'] == feature]['importance'].iloc[0]
        mi_score = mi_results[mi_results['feature'] == feature]['mi_score'].iloc[0]
        f_score = f_test_results[f_test_results['feature'] == feature]['f_score'].iloc[0]
        chi2_score = chi2_results[chi2_results['feature'] == feature]['chi2_score'].iloc[0]
        
        # Calculate combined rank (lower is better)
        combined_rank = (rf_rank + xgb_rank + mi_rank + f_rank + chi2_rank) / 5
        
        combined_rankings.append({
            'feature': feature,
            'rf_rank': rf_rank,
            'xgb_rank': xgb_rank,
            'mi_rank': mi_rank,
            'f_rank': f_rank,
            'chi2_rank': chi2_rank,
            'combined_rank': combined_rank,
            'rf_score': rf_score,
            'xgb_score': xgb_score,
            'mi_score': mi_score,
            'f_score': f_score,
            'chi2_score': chi2_score
        })
    
    combined_df = pd.DataFrame(combined_rankings).sort_values('combined_rank')
    
    print("✅ Feature rankings combined successfully!")
    print(f"✅ Top 5 features overall: {combined_df.head()['feature'].tolist()}")
    
    return combined_df

def create_feature_importance_visualizations(rf_importance, xgb_importance, mi_results, 
                                           chi2_results, combined_df, top_n=15):
    """
    Create comprehensive feature importance visualizations
    """
    print("\n📊 Creating Feature Importance Visualizations...")
    
    fig = plt.figure(figsize=(20, 16))
    
    # 1. Random Forest Feature Importance
    plt.subplot(2, 3, 1)
    top_rf = rf_importance.head(top_n)
    plt.barh(range(len(top_rf)), top_rf['importance'], color='forestgreen', alpha=0.7)
    plt.yticks(range(len(top_rf)), top_rf['feature'])
    plt.xlabel('Importance Score')
    plt.title('Random Forest Feature Importance')
    plt.gca().invert_yaxis()
    plt.grid(True, alpha=0.3)
    
    # 2. XGBoost Feature Importance
    plt.subplot(2, 3, 2)
    top_xgb = xgb_importance.head(top_n)
    plt.barh(range(len(top_xgb)), top_xgb['importance'], color='orange', alpha=0.7)
    plt.yticks(range(len(top_xgb)), top_xgb['feature'])
    plt.xlabel('Importance Score')
    plt.title('XGBoost Feature Importance')
    plt.gca().invert_yaxis()
    plt.grid(True, alpha=0.3)
    
    # 3. Mutual Information Scores
    plt.subplot(2, 3, 3)
    top_mi = mi_results.head(top_n)
    plt.barh(range(len(top_mi)), top_mi['mi_score'], color='purple', alpha=0.7)
    plt.yticks(range(len(top_mi)), top_mi['feature'])
    plt.xlabel('Mutual Information Score')
    plt.title('Mutual Information Feature Selection')
    plt.gca().invert_yaxis()
    plt.grid(True, alpha=0.3)
    
    # 4. Chi-Square Scores
    plt.subplot(2, 3, 4)
    top_chi2 = chi2_results.head(top_n)
    plt.barh(range(len(top_chi2)), top_chi2['chi2_score'], color='red', alpha=0.7)
    plt.yticks(range(len(top_chi2)), top_chi2['feature'])
    plt.xlabel('Chi-Square Score')
    plt.title('Chi-Square Feature Selection')
    plt.gca().invert_yaxis()
    plt.grid(True, alpha=0.3)
    
    # 5. Combined Rankings
    plt.subplot(2, 3, 5)
    top_combined = combined_df.head(top_n)
    colors = plt.cm.viridis(np.linspace(0, 1, len(top_combined)))
    plt.barh(range(len(top_combined)), 1/top_combined['combined_rank'], color=colors, alpha=0.7)
    plt.yticks(range(len(top_combined)), top_combined['feature'])
    plt.xlabel('Combined Importance (1/Rank)')
    plt.title('Combined Feature Ranking')
    plt.gca().invert_yaxis()
    plt.grid(True, alpha=0.3)
    
    # 6. Feature Selection Methods Comparison (Heatmap)
    plt.subplot(2, 3, 6)
    methods_comparison = combined_df.head(10)[['feature', 'rf_rank', 'xgb_rank', 
                                              'mi_rank', 'f_rank', 'chi2_rank']].set_index('feature')
    sns.heatmap(methods_comparison.T, annot=True, cmap='RdYlBu_r', cbar_kws={'label': 'Rank'})
    plt.title('Feature Rankings Across Methods')
    plt.ylabel('Selection Method')
    
    plt.tight_layout()
    plt.savefig('feature_importance_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Feature importance visualizations saved as 'feature_importance_analysis.png'")

def select_final_features(combined_df, n_features_list=[5, 10, 15, 20]):
    """
    Select final features based on combined ranking
    """
    print("\n🎯 Selecting Final Features...")
    
    final_selections = {}
    
    for n in n_features_list:
        if n <= len(combined_df):
            selected = combined_df.head(n)['feature'].tolist()
            final_selections[f'top_{n}'] = selected
            print(f"✅ Top {n} features: {selected}")
    
    return final_selections

def create_reduced_datasets(X, y, final_selections, feature_names):
    """
    Create reduced datasets with selected features
    """
    print("\n💾 Creating Reduced Datasets...")
    
    datasets = {}
    original_data = pd.DataFrame(X, columns=feature_names)
    original_data['target'] = y.values
    
    for selection_name, selected_features in final_selections.items():
        # Create dataset with selected features
        reduced_data = original_data[selected_features + ['target']].copy()
        datasets[selection_name] = reduced_data
        
        print(f"✅ {selection_name}: {len(selected_features)} features, shape: {reduced_data.shape}")
    
    return datasets

def evaluate_feature_sets(X, y, final_selections, feature_names):
    """
    Evaluate model performance with different feature sets
    """
    print("\n🔍 Evaluating Feature Sets Performance...")
    
    results = []
    
    # Baseline with all features
    rf_baseline = RandomForestClassifier(n_estimators=100, random_state=42)
    baseline_scores = cross_val_score(rf_baseline, X, y, cv=5)
    results.append({
        'feature_set': 'all_features',
        'n_features': X.shape[1],
        'cv_mean': baseline_scores.mean(),
        'cv_std': baseline_scores.std()
    })
    
    # Test each feature selection
    for selection_name, selected_features in final_selections.items():
        feature_indices = [feature_names.index(f) for f in selected_features]
        X_selected = X.iloc[:, feature_indices]
        
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        cv_scores = cross_val_score(rf, X_selected, y, cv=5)
        
        results.append({
            'feature_set': selection_name,
            'n_features': len(selected_features),
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std()
        })
        
        print(f"✅ {selection_name}: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    results_df = pd.DataFrame(results).sort_values('cv_mean', ascending=False)
    return results_df

def save_feature_selection_results(datasets, combined_df, final_selections, performance_results):
    """
    Save all feature selection results
    """
    print("\n💾 Saving Feature Selection Results...")
    
    # Save reduced datasets
    for name, dataset in datasets.items():
        filename = f"feature_selected_{name}.csv"
        dataset.to_csv(filename, index=False)
        print(f"✅ Saved {filename}")
    
    # Save feature rankings
    combined_df.to_csv('feature_rankings_combined.csv', index=False)
    print("✅ Saved feature_rankings_combined.csv")
    
    # Save final selections summary
    import json
    summary = {
        'feature_selections': final_selections,
        'performance_comparison': performance_results.to_dict('records'),
        'total_original_features': len(combined_df),
        'recommended_feature_set': performance_results.iloc[1]['feature_set']  # Best performing (excluding all features)
    }
    
    with open('feature_selection_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("✅ Saved feature_selection_summary.json")

def main():
    """
    Main function to execute feature selection analysis
    """
    print("🚀 Starting Feature Selection Analysis")
    print("=" * 60)
    
    # 1. Load preprocessed data
    data_file = "model_ready_data.csv"
    data = load_preprocessed_data(data_file)
    
    if data is None:
        return
    
    # 2. Prepare data
    X, y, feature_names = prepare_data_for_selection(data)
    
    # 3. Random Forest Feature Importance
    rf_importance, rf_model = random_forest_feature_importance(X, y, feature_names)
    
    # 4. XGBoost Feature Importance
    xgb_importance, xgb_model = xgboost_feature_importance(X, y, feature_names)
    
    # 5. Recursive Feature Elimination
    rfe_results, rfe_selected, rfe_model = recursive_feature_elimination(X, y, feature_names)
    
    # 6. Chi-Square Feature Selection
    chi2_results, chi2_selected, chi2_selector = chi_square_feature_selection(X, y, feature_names)
    
    # 7. Mutual Information Selection
    mi_results, mi_selected = mutual_information_selection(X, y, feature_names)
    
    # 8. Univariate Statistical Tests
    f_test_results, f_significant = univariate_statistical_tests(X, y, feature_names)
    
    # 9. Combine all rankings
    combined_df = combine_feature_rankings(rf_importance, xgb_importance, mi_results, 
                                         f_test_results, chi2_results)
    
    # 10. Create visualizations
    create_feature_importance_visualizations(rf_importance, xgb_importance, mi_results,
                                           chi2_results, combined_df)
    
    # 11. Select final features
    final_selections = select_final_features(combined_df)
    
    # 12. Create reduced datasets
    reduced_datasets = create_reduced_datasets(X, y, final_selections, feature_names)
    
    # 13. Evaluate feature sets
    performance_results = evaluate_feature_sets(X, y, final_selections, feature_names)
    
    # 14. Save results
    save_feature_selection_results(reduced_datasets, combined_df, final_selections, performance_results)
    
    print("\n🎉 Feature Selection Analysis Complete!")
    print("=" * 60)
    print("📋 Deliverables Created:")
    print("   ✔ Reduced datasets with selected features (5, 10, 15, 20 features)")
    print("   ✔ Feature importance ranking visualizations")
    print("   ✔ Comprehensive feature analysis report")
    print("   ✔ Performance comparison of different feature sets")
    
    print(f"\n🏆 Best performing feature set: {performance_results.iloc[1]['feature_set']}")
    print(f"    Features: {performance_results.iloc[1]['n_features']}")
    print(f"    CV Score: {performance_results.iloc[1]['cv_mean']:.4f}")
    
    return reduced_datasets, combined_df, final_selections

if __name__ == "__main__":
    # Execute the main analysis
    results = main()