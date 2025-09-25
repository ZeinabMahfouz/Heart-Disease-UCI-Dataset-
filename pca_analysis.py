# PCA Dimensionality Reduction Analysis
# Step 2: Principal Component Analysis for Heart Disease Dataset

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_preprocessed_data(file_path):
    """
    Load the preprocessed data from step 1
    """
    try:
        # Try different common file formats
        if file_path.endswith('.csv'):
            data = pd.read_csv(file_path)
        elif file_path.endswith('.pkl'):
            data = pd.read_pickle(file_path)
        elif file_path.endswith('.xlsx'):
            data = pd.read_excel(file_path)
        else:
            # Default to CSV
            data = pd.read_csv(file_path)
        
        print(f"✅ Data loaded successfully!")
        print(f"Dataset shape: {data.shape}")
        return data
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None

def prepare_features_target(data, target_column='target'):
    """
    Separate features and target variable
    """
    if target_column in data.columns:
        X = data.drop(target_column, axis=1)
        y = data[target_column]
        print(f"✅ Features: {X.shape[1]} columns")
        print(f"✅ Target variable separated")
    else:
        # If no target column specified, assume last column is target
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]
        print(f"✅ Assumed last column as target")
        print(f"✅ Features: {X.shape[1]} columns")
    
    return X, y

def apply_pca_analysis(X, n_components=None):
    """
    Apply PCA and determine optimal number of components
    """
    # Ensure data is standardized (should be done in step 1, but double-check)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # If n_components not specified, use all components for analysis
    if n_components is None:
        n_components = min(X.shape[0], X.shape[1])
    
    # Apply PCA
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    
    print(f"✅ PCA applied with {n_components} components")
    print(f"✅ Original shape: {X.shape}")
    print(f"✅ PCA transformed shape: {X_pca.shape}")
    
    return pca, X_pca, X_scaled

def determine_optimal_components(pca, variance_threshold=0.95):
    """
    Determine optimal number of components based on explained variance
    """
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance_ratio)
    
    # Find number of components for desired variance threshold
    n_components_95 = np.argmax(cumulative_variance >= variance_threshold) + 1
    n_components_90 = np.argmax(cumulative_variance >= 0.90) + 1
    n_components_85 = np.argmax(cumulative_variance >= 0.85) + 1
    
    print(f"\n📊 Optimal Components Analysis:")
    print(f"Components for 85% variance: {n_components_85}")
    print(f"Components for 90% variance: {n_components_90}")
    print(f"Components for 95% variance: {n_components_95}")
    print(f"Total variance with all components: {cumulative_variance[-1]:.4f}")
    
    return {
        '85%': n_components_85,
        '90%': n_components_90,
        '95%': n_components_95,
        'cumulative_variance': cumulative_variance,
        'individual_variance': explained_variance_ratio
    }

def visualize_pca_results(pca, X_pca, y, variance_info):
    """
    Create comprehensive visualizations for PCA results
    """
    fig = plt.figure(figsize=(20, 15))
    
    # 1. Scree Plot - Individual Variance per Component
    plt.subplot(2, 3, 1)
    components = range(1, len(pca.explained_variance_ratio_) + 1)
    plt.bar(components[:15], pca.explained_variance_ratio_[:15], alpha=0.7, color='skyblue')
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.title('Scree Plot - Variance per Component (First 15)')
    plt.xticks(components[:15])
    plt.grid(True, alpha=0.3)
    
    # 2. Cumulative Variance Plot
    plt.subplot(2, 3, 2)
    cumulative_var = variance_info['cumulative_variance']
    plt.plot(components, cumulative_var, 'bo-', linewidth=2, markersize=6)
    plt.axhline(y=0.85, color='r', linestyle='--', alpha=0.7, label='85% Variance')
    plt.axhline(y=0.90, color='orange', linestyle='--', alpha=0.7, label='90% Variance')
    plt.axhline(y=0.95, color='g', linestyle='--', alpha=0.7, label='95% Variance')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance Ratio')
    plt.title('Cumulative Explained Variance')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 3. PCA Scatter Plot (First 2 Components)
    plt.subplot(2, 3, 3)
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', alpha=0.7)
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    plt.title('PCA Scatter Plot (PC1 vs PC2)')
    plt.colorbar(scatter, label='Target')
    plt.grid(True, alpha=0.3)
    
    # 4. Feature Contribution to PC1 and PC2
    plt.subplot(2, 3, 4)
    feature_names = [f'Feature_{i}' for i in range(len(pca.components_[0]))]
    if len(feature_names) > 15:
        # Show only top contributing features
        pc1_contributions = np.abs(pca.components_[0])
        top_features_idx = np.argsort(pc1_contributions)[-10:]
        plt.barh(range(10), pc1_contributions[top_features_idx])
        plt.yticks(range(10), [feature_names[i] for i in top_features_idx])
    else:
        plt.barh(range(len(feature_names)), np.abs(pca.components_[0]))
        plt.yticks(range(len(feature_names)), feature_names)
    plt.xlabel('Absolute Contribution')
    plt.title('Feature Contributions to PC1')
    plt.grid(True, alpha=0.3)
    
    # 5. Variance Explained by Top 10 Components (Bar Chart)
    plt.subplot(2, 3, 5)
    top_10_components = min(10, len(pca.explained_variance_ratio_))
    plt.bar(range(1, top_10_components + 1), 
            pca.explained_variance_ratio_[:top_10_components], 
            alpha=0.7, color='lightcoral')
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.title('Top 10 Components - Variance Explained')
    plt.xticks(range(1, top_10_components + 1))
    plt.grid(True, alpha=0.3)
    
    # 6. 3D Scatter Plot (First 3 Components)
    ax = fig.add_subplot(2, 3, 6, projection='3d')
    scatter_3d = ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], 
                          c=y, cmap='viridis', alpha=0.6)
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    ax.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]:.1%})')
    ax.set_title('3D PCA Plot (First 3 Components)')
    
    plt.tight_layout()
    plt.savefig('pca_analysis_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ PCA visualizations saved as 'pca_analysis_results.png'")

def create_pca_transformed_datasets(pca, X_scaled, y, variance_info):
    """
    Create datasets with different numbers of components
    """
    datasets = {}
    
    for threshold, n_components in variance_info.items():
        if threshold in ['85%', '90%', '95%']:
            # Create PCA with specific number of components
            pca_subset = PCA(n_components=n_components)
            X_pca_subset = pca_subset.fit_transform(X_scaled)
            
            # Create DataFrame
            column_names = [f'PC{i+1}' for i in range(n_components)]
            df_pca = pd.DataFrame(X_pca_subset, columns=column_names)
            df_pca['target'] = y.values
            
            datasets[threshold] = df_pca
            
            print(f"✅ Dataset created for {threshold} variance: {df_pca.shape}")
    
    return datasets

def save_pca_results(datasets, pca, variance_info):
    """
    Save PCA results to files
    """
    # Save the main PCA-transformed datasets
    for threshold, dataset in datasets.items():
        filename = f"pca_dataset_{threshold.replace('%', 'pct')}.csv"
        dataset.to_csv(filename, index=False)
        print(f"✅ Saved {filename}")
    
    # Convert numpy data to native Python types
    pca_info = {
        'n_components': int(len(pca.explained_variance_ratio_)),
        'explained_variance_ratio': [float(x) for x in pca.explained_variance_ratio_],
        'explained_variance': [float(x) for x in pca.explained_variance_],
        'components': [[float(val) for val in row] for row in pca.components_],
        'optimal_components': {str(k): int(v) for k, v in variance_info.items() 
                               if k in ['85%', '90%', '95%']}
    }
    
    import json
    with open('pca_analysis_summary.json', 'w') as f:
        json.dump(pca_info, f, indent=2)
    
    print("✅ PCA analysis summary saved as 'pca_analysis_summary.json'")
def main():
    """
    Main function to execute PCA analysis
    """
    print("🚀 Starting PCA Dimensionality Reduction Analysis")
    print("=" * 60)
    
    # 1. Load preprocessed data
    data_file = "model_ready_data.csv"  # Adjust filename as needed
    data = load_preprocessed_data(data_file)
    
    if data is None:
        print("❌ Please ensure your preprocessed data file exists and try again.")
        return
    
    # 2. Prepare features and target
    X, y = prepare_features_target(data)
    
    # 3. Apply PCA
    pca, X_pca, X_scaled = apply_pca_analysis(X)
    
    # 4. Determine optimal components
    variance_info = determine_optimal_components(pca)
    
    # 5. Create visualizations
    print("\n📊 Creating PCA visualizations...")
    visualize_pca_results(pca, X_pca, y, variance_info)
    
    # 6. Create transformed datasets
    print("\n💾 Creating PCA-transformed datasets...")
    pca_datasets = create_pca_transformed_datasets(pca, X_scaled, y, variance_info)
    
    # 7. Save results
    print("\n💾 Saving PCA results...")
    save_pca_results(pca_datasets, pca, variance_info)
    
    print("\n🎉 PCA Analysis Complete!")
    print("=" * 60)
    print("📋 Deliverables Created:")
    print("   ✔ PCA-transformed datasets (85%, 90%, 95% variance)")
    print("   ✔ Comprehensive PCA visualizations")
    print("   ✔ Variance analysis and component selection")
    print("   ✔ PCA analysis summary (JSON)")
    
    # Return key results for further analysis
    return pca_datasets, pca, variance_info

if __name__ == "__main__":
    # Execute the main analysis
    results = main()