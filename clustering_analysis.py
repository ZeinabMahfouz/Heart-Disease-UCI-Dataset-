import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score, calinski_harabasz_score
from sklearn.manifold import TSNE
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist
import joblib
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_dataset_for_clustering(file_path=None):
    """
    Load dataset for clustering analysis
    Priority: Feature Selected > PCA > Original preprocessed data
    """
    datasets_to_try = [
        "feature_selected_top_10.csv",  # From step 2.3 - BEST for clustering
        "feature_selected_top_15.csv",  # From step 2.3 - Alternative
        "pca_dataset_90pct.csv",        # From step 2.2 - Good for visualization
        "pca_dataset_95pct.csv",        # From step 2.2 - More features
        "model_ready_data.csv"          # From step 2.1 - Original features
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
    print("   - pca_dataset_Xpct.csv (from step 2.2 - Good for visualization)")
    print("   - model_ready_data.csv (from step 2.1)")
    return None, None

def prepare_clustering_data(data, target_column='target'):
    """
    Prepare features for clustering analysis
    """
    if target_column in data.columns:
        X = data.drop(target_column, axis=1)
        y = data[target_column]
    else:
        # Assume last column is target
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]
    
    feature_names = X.columns.tolist()
    
    print(f"✅ Features for clustering: {X.shape[1]} columns")
    print(f"✅ True labels available for comparison: {y.value_counts().to_dict()}")
    print(f"✅ Feature names: {feature_names}")
    
    # Check if data needs scaling
    if not all(X.std() < 2):  # If standard deviation > 2, likely needs scaling
        print("🔧 Data appears to need scaling - will apply StandardScaler")
        needs_scaling = True
    else:
        print("✅ Data appears to be already scaled")
        needs_scaling = False
    
    return X, y, feature_names, needs_scaling

def scale_features_for_clustering(X, needs_scaling=True):
    """
    Scale features for clustering if needed
    """
    if needs_scaling:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
        print("✅ Features scaled using StandardScaler")
        return X_scaled, scaler
    else:
        print("✅ Using original features (already scaled)")
        return X, None

def elbow_method_analysis(X_scaled, max_k=10):
    """
    Perform elbow method analysis to determine optimal K for K-Means
    """
    print(f"\nPerforming Elbow Method Analysis (K=1 to {max_k})...")

    # Ensure input is a numeric numpy array
    X_scaled = np.asarray(X_scaled, dtype=float)

    wcss = []
    silhouette_scores = []
    calinski_scores = []
    k_range = range(1, max_k + 1)

    for k in k_range:
        if k == 1:
            # WCSS when k=1 = total variance (distance to overall mean)
            overall_mean = np.mean(X_scaled, axis=0)
            wcss_1 = np.sum((X_scaled - overall_mean) ** 2)
            wcss.append(wcss_1)
            silhouette_scores.append(0)  # not defined for k=1
            calinski_scores.append(0)    # not defined for k=1
        else:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(X_scaled)
            wcss.append(kmeans.inertia_)

            sil_score = silhouette_score(X_scaled, kmeans.labels_)
            silhouette_scores.append(sil_score)

            calinski = calinski_harabasz_score(X_scaled, kmeans.labels_)
            calinski_scores.append(calinski)

        print(f"K={k}: WCSS={wcss[-1]:.2f}, Silhouette={silhouette_scores[-1]:.3f}, Calinski={calinski_scores[-1]:.2f}")

    # Heuristic for elbow method
    wcss_diff = np.diff(wcss)
    wcss_diff2 = np.diff(wcss_diff)
    if len(wcss_diff2) > 0:
        elbow_k = np.argmax(wcss_diff2) + 2  # +2 because of double differencing
    else:
        elbow_k = 3  # fallback

    # Best K based on silhouette score (ignoring k=1)
    best_sil_k = k_range[np.argmax(silhouette_scores[1:]) + 1]

    # Best K based on Calinski-Harabasz score (ignoring k=1)
    best_cal_k = k_range[np.argmax(calinski_scores[1:]) + 1]

    print("\nOptimal K Analysis:")
    print(f"  Elbow Method suggests K = {elbow_k}")
    print(f"  Best Silhouette Score K = {best_sil_k} (Score: {silhouette_scores[best_sil_k-1]:.3f})")
    print(f"  Best Calinski-Harabasz K = {best_cal_k} (Score: {calinski_scores[best_cal_k-1]:.2f})")

    return {
        "k_range": list(k_range),
        "wcss": wcss,
        "silhouette_scores": silhouette_scores,
        "calinski_scores": calinski_scores,
        "elbow_k": elbow_k,
        "best_silhouette_k": best_sil_k,
        "best_calinski_k": best_cal_k
    }

def perform_kmeans_clustering(X_scaled, optimal_k_info):
    """
    Perform K-Means clustering with different K values
    """
    print(f"\n🎯 Performing K-Means Clustering...")
    
    # Test different K values based on analysis
    k_values_to_test = [
        optimal_k_info['elbow_k'],
        optimal_k_info['best_silhouette_k'],
        optimal_k_info['best_calinski_k'],
        2  # Always test k=2 for binary classification comparison
    ]
    
    # Remove duplicates and sort
    k_values_to_test = sorted(list(set(k_values_to_test)))
    
    kmeans_results = {}
    
    for k in k_values_to_test:
        print(f"\n🔄 K-Means with K={k}...")
        
        # Perform K-Means clustering
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X_scaled)
        
        # Calculate metrics
        silhouette = silhouette_score(X_scaled, cluster_labels)
        calinski = calinski_harabasz_score(X_scaled, cluster_labels)
        
        kmeans_results[k] = {
            'model': kmeans,
            'labels': cluster_labels,
            'silhouette_score': silhouette,
            'calinski_score': calinski,
            'inertia': kmeans.inertia_,
            'cluster_centers': kmeans.cluster_centers_
        }
        
        print(f"✅ K={k} - Silhouette: {silhouette:.3f}, Calinski-Harabasz: {calinski:.2f}")
        print(f"   Cluster distribution: {np.bincount(cluster_labels)}")
    
    # Select best K based on silhouette score
    best_k = max(kmeans_results.keys(), key=lambda k: kmeans_results[k]['silhouette_score'])
    print(f"\n🏆 Best K-Means model: K={best_k} (Silhouette Score: {kmeans_results[best_k]['silhouette_score']:.3f})")
    
    return kmeans_results, best_k

def perform_hierarchical_clustering(X_scaled, n_clusters_range=[2, 3, 4, 5]):
    """
    Perform Hierarchical Clustering with dendrogram analysis
    """
    print(f"\n🌳 Performing Hierarchical Clustering...")
    
    # Calculate linkage matrix for dendrogram
    print("📊 Calculating linkage matrix for dendrogram...")
    linkage_methods = ['ward', 'complete', 'average', 'single']
    linkage_matrices = {}
    
    for method in linkage_methods:
        if method == 'ward':
            # Ward requires euclidean distance
            Z = linkage(X_scaled, method=method, metric='euclidean')
        else:
            Z = linkage(X_scaled, method=method, metric='euclidean')
        linkage_matrices[method] = Z
        print(f"✅ {method.title()} linkage calculated")
    
    # Perform hierarchical clustering with different number of clusters
    hierarchical_results = {}
    
    # Use Ward method as default (usually performs best)
    ward_linkage = linkage_matrices['ward']
    
    for n_clusters in n_clusters_range:
        print(f"\n🔄 Hierarchical Clustering with {n_clusters} clusters...")
        
        # Perform hierarchical clustering
        hierarchical = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
        cluster_labels = hierarchical.fit_predict(X_scaled)
        
        # Calculate metrics
        silhouette = silhouette_score(X_scaled, cluster_labels)
        calinski = calinski_harabasz_score(X_scaled, cluster_labels)
        
        hierarchical_results[n_clusters] = {
            'model': hierarchical,
            'labels': cluster_labels,
            'silhouette_score': silhouette,
            'calinski_score': calinski
        }
        
        print(f"✅ {n_clusters} clusters - Silhouette: {silhouette:.3f}, Calinski-Harabasz: {calinski:.2f}")
        print(f"   Cluster distribution: {np.bincount(cluster_labels)}")
    
    # Select best number of clusters based on silhouette score
    best_n_clusters = max(hierarchical_results.keys(), 
                         key=lambda n: hierarchical_results[n]['silhouette_score'])
    print(f"\n🏆 Best Hierarchical model: {best_n_clusters} clusters (Silhouette: {hierarchical_results[best_n_clusters]['silhouette_score']:.3f})")
    
    return hierarchical_results, best_n_clusters, linkage_matrices

def compare_clusters_with_labels(kmeans_results, hierarchical_results, y, best_kmeans_k, best_hierarchical_n):
    """
    Compare clustering results with actual disease labels
    """
    print(f"\n🔍 Comparing Clusters with Actual Disease Labels...")
    
    comparison_results = {}
    
    # Compare K-Means results
    print(f"\n📊 K-Means Comparison (K={best_kmeans_k}):")
    kmeans_labels = kmeans_results[best_kmeans_k]['labels']
    
    # Calculate similarity metrics
    kmeans_ari = adjusted_rand_score(y, kmeans_labels)
    kmeans_nmi = normalized_mutual_info_score(y, kmeans_labels)
    
    comparison_results['kmeans'] = {
        'adjusted_rand_score': kmeans_ari,
        'normalized_mutual_info': kmeans_nmi,
        'cluster_labels': kmeans_labels
    }
    
    print(f"✅ K-Means vs True Labels:")
    print(f"   Adjusted Rand Index: {kmeans_ari:.3f}")
    print(f"   Normalized Mutual Information: {kmeans_nmi:.3f}")
    
    # Compare Hierarchical results
    print(f"\n📊 Hierarchical Comparison ({best_hierarchical_n} clusters):")
    hierarchical_labels = hierarchical_results[best_hierarchical_n]['labels']
    
    hierarchical_ari = adjusted_rand_score(y, hierarchical_labels)
    hierarchical_nmi = normalized_mutual_info_score(y, hierarchical_labels)
    
    comparison_results['hierarchical'] = {
        'adjusted_rand_score': hierarchical_ari,
        'normalized_mutual_info': hierarchical_nmi,
        'cluster_labels': hierarchical_labels
    }
    
    print(f"✅ Hierarchical vs True Labels:")
    print(f"   Adjusted Rand Index: {hierarchical_ari:.3f}")
    print(f"   Normalized Mutual Information: {hierarchical_nmi:.3f}")
    
    # Create confusion matrices
    print(f"\n📋 Cluster vs Label Distribution Analysis:")
    
    # K-Means confusion matrix
    kmeans_crosstab = pd.crosstab(kmeans_labels, y, margins=True)
    print(f"\n🎯 K-Means Cluster vs True Label Distribution:")
    print(kmeans_crosstab)
    
    # Hierarchical confusion matrix
    hierarchical_crosstab = pd.crosstab(hierarchical_labels, y, margins=True)
    print(f"\n🌳 Hierarchical Cluster vs True Label Distribution:")
    print(hierarchical_crosstab)
    
    comparison_results['kmeans_crosstab'] = kmeans_crosstab
    comparison_results['hierarchical_crosstab'] = hierarchical_crosstab
    
    return comparison_results

def create_clustering_visualizations(X_scaled, optimal_k_info, kmeans_results, hierarchical_results, 
                                   linkage_matrices, comparison_results, y, best_kmeans_k, best_hierarchical_n):
    """
    Create comprehensive clustering visualizations
    """
    print(f"\n📊 Creating Clustering Visualizations...")
    
    fig = plt.figure(figsize=(24, 20))
    
    # 1. Elbow Method Plot
    plt.subplot(3, 4, 1)
    k_range = optimal_k_info['k_range']
    wcss = optimal_k_info['wcss']
    plt.plot(k_range, wcss, 'bo-', linewidth=2, markersize=8)
    plt.axvline(x=optimal_k_info['elbow_k'], color='r', linestyle='--', alpha=0.7, 
                label=f'Elbow K={optimal_k_info["elbow_k"]}')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('WCSS (Within-Cluster Sum of Squares)')
    plt.title('Elbow Method for Optimal K')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. Silhouette Score Plot
    plt.subplot(3, 4, 2)
    sil_scores = optimal_k_info['silhouette_scores'][1:]  # Exclude k=1
    k_range_sil = list(k_range)[1:]  # Exclude k=1
    plt.plot(k_range_sil, sil_scores, 'go-', linewidth=2, markersize=8)
    plt.axvline(x=optimal_k_info['best_silhouette_k'], color='r', linestyle='--', alpha=0.7,
                label=f'Best K={optimal_k_info["best_silhouette_k"]}')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score Analysis')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 3. Dendrogram (Ward method)
    plt.subplot(3, 4, 3)
    ward_linkage = linkage_matrices['ward']
    dendrogram(ward_linkage, truncate_mode='level', p=5)
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('Sample Index or Cluster Size')
    plt.ylabel('Distance')
    
    # 4. Calinski-Harabasz Score
    plt.subplot(3, 4, 4)
    cal_scores = optimal_k_info['calinski_scores'][1:]  # Exclude k=1
    plt.plot(k_range_sil, cal_scores, 'mo-', linewidth=2, markersize=8)
    plt.axvline(x=optimal_k_info['best_calinski_k'], color='r', linestyle='--', alpha=0.7,
                label=f'Best K={optimal_k_info["best_calinski_k"]}')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Calinski-Harabasz Score')
    plt.title('Calinski-Harabasz Score Analysis')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # For 2D visualization, use PCA if we have more than 2 features
    if X_scaled.shape[1] > 2:
        # Apply PCA for 2D visualization
        pca_vis = PCA(n_components=2, random_state=42)
        X_pca = pca_vis.fit_transform(X_scaled)
        x_col, y_col = 0, 1
        x_label = f'PC1 ({pca_vis.explained_variance_ratio_[0]:.1%} variance)'
        y_label = f'PC2 ({pca_vis.explained_variance_ratio_[1]:.1%} variance)'
        title_suffix = '(PCA 2D Projection)'
    else:
        X_pca = X_scaled.values
        x_col, y_col = 0, 1
        x_label, y_label = X_scaled.columns[0], X_scaled.columns[1]
        title_suffix = ''
    
    # 5. K-Means Clustering Visualization
    plt.subplot(3, 4, 5)
    kmeans_labels = kmeans_results[best_kmeans_k]['labels']
    colors = plt.cm.Set1(np.linspace(0, 1, best_kmeans_k))
    for i in range(best_kmeans_k):
        mask = kmeans_labels == i
        plt.scatter(X_pca[mask, x_col], X_pca[mask, y_col], 
                   c=[colors[i]], label=f'Cluster {i}', alpha=0.7, s=50)
    
    # Plot centroids if available and in 2D
    if X_scaled.shape[1] == 2:
        centroids = kmeans_results[best_kmeans_k]['cluster_centers']
        plt.scatter(centroids[:, 0], centroids[:, 1], 
                   c='red', marker='x', s=200, linewidths=3, label='Centroids')
    
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(f'K-Means Clustering (K={best_kmeans_k}) {title_suffix}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 6. Hierarchical Clustering Visualization
    plt.subplot(3, 4, 6)
    hierarchical_labels = hierarchical_results[best_hierarchical_n]['labels']
    colors_hier = plt.cm.Set2(np.linspace(0, 1, best_hierarchical_n))
    for i in range(best_hierarchical_n):
        mask = hierarchical_labels == i
        plt.scatter(X_pca[mask, x_col], X_pca[mask, y_col], 
                   c=[colors_hier[i]], label=f'Cluster {i}', alpha=0.7, s=50)
    
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(f'Hierarchical Clustering ({best_hierarchical_n} clusters) {title_suffix}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 7. True Labels Visualization
    plt.subplot(3, 4, 7)
    colors_true = ['red' if label == 1 else 'blue' for label in y]
    scatter = plt.scatter(X_pca[:, x_col], X_pca[:, y_col], c=colors_true, alpha=0.7, s=50)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(f'True Labels {title_suffix}')
    # Create custom legend
    import matplotlib.patches as mpatches
    blue_patch = mpatches.Patch(color='blue', label='No Disease')
    red_patch = mpatches.Patch(color='red', label='Disease')
    plt.legend(handles=[blue_patch, red_patch])
    plt.grid(True, alpha=0.3)
    
    # 8. Cluster Comparison Heatmap (K-Means)
    plt.subplot(3, 4, 8)
    kmeans_crosstab = comparison_results['kmeans_crosstab'].iloc[:-1, :-1]  # Remove margins
    sns.heatmap(kmeans_crosstab, annot=True, fmt='d', cmap='Blues')
    plt.title('K-Means vs True Labels')
    plt.xlabel('True Labels')
    plt.ylabel('K-Means Clusters')
    
    # 9. Method Comparison (ARI and NMI)
    plt.subplot(3, 4, 9)
    methods = ['K-Means', 'Hierarchical']
    ari_scores = [comparison_results['kmeans']['adjusted_rand_score'],
                  comparison_results['hierarchical']['adjusted_rand_score']]
    nmi_scores = [comparison_results['kmeans']['normalized_mutual_info'],
                  comparison_results['hierarchical']['normalized_mutual_info']]
    
    x = np.arange(len(methods))
    width = 0.35
    
    plt.bar(x - width/2, ari_scores, width, label='Adjusted Rand Index', alpha=0.7)
    plt.bar(x + width/2, nmi_scores, width, label='Normalized Mutual Info', alpha=0.7)
    
    plt.xlabel('Clustering Method')
    plt.ylabel('Score')
    plt.title('Clustering vs True Labels Comparison')
    plt.xticks(x, methods)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 10. Silhouette Scores Comparison
    plt.subplot(3, 4, 10)
    methods_sil = ['K-Means', 'Hierarchical']
    silhouette_scores_comp = [kmeans_results[best_kmeans_k]['silhouette_score'],
                             hierarchical_results[best_hierarchical_n]['silhouette_score']]
    
    bars = plt.bar(methods_sil, silhouette_scores_comp, alpha=0.7, 
                   color=['skyblue', 'lightcoral'])
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score Comparison')
    plt.ylim(0, max(silhouette_scores_comp) * 1.1)
    
    # Add value labels on bars
    for bar, score in zip(bars, silhouette_scores_comp):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{score:.3f}', ha='center', va='bottom')
    
    plt.grid(True, alpha=0.3)
    
    # 11. Cluster Size Distribution
    plt.subplot(3, 4, 11)
    kmeans_cluster_sizes = np.bincount(kmeans_labels)
    hierarchical_cluster_sizes = np.bincount(hierarchical_labels)
    
    x = np.arange(max(len(kmeans_cluster_sizes), len(hierarchical_cluster_sizes)))
    width = 0.35
    
    # Pad shorter array with zeros
    if len(kmeans_cluster_sizes) < len(x):
        kmeans_cluster_sizes = np.pad(kmeans_cluster_sizes, (0, len(x) - len(kmeans_cluster_sizes)))
    if len(hierarchical_cluster_sizes) < len(x):
        hierarchical_cluster_sizes = np.pad(hierarchical_cluster_sizes, (0, len(x) - len(hierarchical_cluster_sizes)))
    
    plt.bar(x - width/2, kmeans_cluster_sizes[:len(x)], width, label='K-Means', alpha=0.7)
    plt.bar(x + width/2, hierarchical_cluster_sizes[:len(x)], width, label='Hierarchical', alpha=0.7)
    
    plt.xlabel('Cluster ID')
    plt.ylabel('Number of Samples')
    plt.title('Cluster Size Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 12. Feature Importance in Clustering (if possible)
    plt.subplot(3, 4, 12)
    # For K-means, show distance to centroids for each feature
    if X_scaled.shape[1] <= 10:  # Only show if reasonable number of features
        if best_kmeans_k == 2:  # Binary clustering
            centroid_diff = np.abs(kmeans_results[best_kmeans_k]['cluster_centers'][0] - 
                                 kmeans_results[best_kmeans_k]['cluster_centers'][1])
            feature_importance = centroid_diff / np.sum(centroid_diff)
            
            plt.barh(range(len(X_scaled.columns)), feature_importance, alpha=0.7)
            plt.yticks(range(len(X_scaled.columns)), X_scaled.columns)
            plt.xlabel('Feature Importance (Normalized)')
            plt.title('Feature Importance in K-Means Clustering')
            plt.grid(True, alpha=0.3)
        else:
            plt.text(0.5, 0.5, 'Feature importance\nanalysis available\nonly for K=2', 
                    ha='center', va='center', transform=plt.gca().transAxes, fontsize=12)
            plt.title('Feature Importance Analysis')
    else:
        plt.text(0.5, 0.5, f'Too many features ({X_scaled.shape[1]})\nfor visualization', 
                ha='center', va='center', transform=plt.gca().transAxes, fontsize=12)
        plt.title('Feature Importance Analysis')
    
    plt.tight_layout()
    plt.savefig('clustering_analysis_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Clustering visualizations saved as 'clustering_analysis_results.png'")

def save_clustering_results(kmeans_results, hierarchical_results, optimal_k_info, 
                          comparison_results, best_kmeans_k, best_hierarchical_n, dataset_source):
    """
    Save clustering models and results
    """
    print("\n💾 Saving Clustering Results...")
    
    # Create directory for models
    import os
    models_dir = "clustering_models/"
    os.makedirs(models_dir, exist_ok=True)
    
    # Save best K-means model
    joblib.dump(kmeans_results[best_kmeans_k]['model'], 
                f"{models_dir}kmeans_model_k{best_kmeans_k}.pkl")
    print(f"✅ Saved K-means model (K={best_kmeans_k})")
    
    # Save best hierarchical model
    joblib.dump(hierarchical_results[best_hierarchical_n]['model'], 
                f"{models_dir}hierarchical_model_{best_hierarchical_n}clusters.pkl")
    print(f"✅ Saved Hierarchical model ({best_hierarchical_n} clusters)")
    
    # Save clustering results summary
    import json
    
    # Convert numpy types for JSON serialization
    results_summary = {
        'dataset_source': dataset_source,
        'optimal_k_analysis': {
            'elbow_method_k': int(optimal_k_info['elbow_k']),
            'best_silhouette_k': int(optimal_k_info['best_silhouette_k']),
            'best_calinski_k': int(optimal_k_info['best_calinski_k'])
        },
        'best_kmeans': {
            'k': int(best_kmeans_k),
            'silhouette_score': float(kmeans_results[best_kmeans_k]['silhouette_score']),
            'calinski_score': float(kmeans_results[best_kmeans_k]['calinski_score']),
            'inertia': float(kmeans_results[best_kmeans_k]['inertia'])
        },
        'best_hierarchical': {
            'n_clusters': int(best_hierarchical_n),
            'silhouette_score': float(hierarchical_results[best_hierarchical_n]['silhouette_score']),
            'calinski_score': float(hierarchical_results[best_hierarchical_n]['calinski_score'])
        },
        'comparison_with_true_labels': {
            'kmeans': {
                'adjusted_rand_score': float(comparison_results['kmeans']['adjusted_rand_score']),
                'normalized_mutual_info': float(comparison_results['kmeans']['normalized_mutual_info'])
            },
            'hierarchical': {
                'adjusted_rand_score': float(comparison_results['hierarchical']['adjusted_rand_score']),
                'normalized_mutual_info': float(comparison_results['hierarchical']['normalized_mutual_info'])
            }
        }
    }
    
    with open('clustering_results_summary.json', 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print("✅ Saved clustering_results_summary.json")
    
    # Save cluster labels
    cluster_labels_df = pd.DataFrame({
        'kmeans_labels': kmeans_results[best_kmeans_k]['labels'],
        'hierarchical_labels': hierarchical_results[best_hierarchical_n]['labels']
    })
    cluster_labels_df.to_csv('cluster_labels.csv', index=False)
    print("✅ Saved cluster_labels.csv")
    
    # Save detailed performance metrics
    performance_data = []
    
    # K-means performance for all tested K values
    for k, result in kmeans_results.items():
        performance_data.append({
            'method': 'K-Means',
            'n_clusters': k,
            'silhouette_score': result['silhouette_score'],
            'calinski_score': result['calinski_score'],
            'inertia': result['inertia']
        })
    
    # Hierarchical performance for all tested cluster numbers
    for n, result in hierarchical_results.items():
        performance_data.append({
            'method': 'Hierarchical',
            'n_clusters': n,
            'silhouette_score': result['silhouette_score'],
            'calinski_score': result['calinski_score'],
            'inertia': None  # Not applicable for hierarchical
        })
    
    performance_df = pd.DataFrame(performance_data)
    performance_df.to_csv('clustering_performance_comparison.csv', index=False)
    print("✅ Saved clustering_performance_comparison.csv")

def create_additional_analysis(X_scaled, kmeans_results, hierarchical_results, y, 
                              best_kmeans_k, best_hierarchical_n):
    """
    Create additional analysis including t-SNE visualization and cluster profiling
    """
    print("\n🔍 Creating Additional Analysis...")
    
    # t-SNE visualization for better clustering visualization
    if X_scaled.shape[0] <= 1000:  # t-SNE is computationally expensive
        print("📊 Creating t-SNE visualization...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, X_scaled.shape[0]//4))
        X_tsne = tsne.fit_transform(X_scaled)
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # K-means t-SNE plot
        kmeans_labels = kmeans_results[best_kmeans_k]['labels']
        colors = plt.cm.Set1(np.linspace(0, 1, best_kmeans_k))
        
        for i in range(best_kmeans_k):
            mask = kmeans_labels == i
            axes[0].scatter(X_tsne[mask, 0], X_tsne[mask, 1], 
                           c=[colors[i]], label=f'Cluster {i}', alpha=0.7, s=50)
        
        axes[0].set_title(f'K-Means Clustering (K={best_kmeans_k}) - t-SNE')
        axes[0].set_xlabel('t-SNE 1')
        axes[0].set_ylabel('t-SNE 2')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Hierarchical t-SNE plot
        hierarchical_labels = hierarchical_results[best_hierarchical_n]['labels']
        colors_hier = plt.cm.Set2(np.linspace(0, 1, best_hierarchical_n))
        
        for i in range(best_hierarchical_n):
            mask = hierarchical_labels == i
            axes[1].scatter(X_tsne[mask, 0], X_tsne[mask, 1], 
                           c=[colors_hier[i]], label=f'Cluster {i}', alpha=0.7, s=50)
        
        axes[1].set_title(f'Hierarchical Clustering ({best_hierarchical_n} clusters) - t-SNE')
        axes[1].set_xlabel('t-SNE 1')
        axes[1].set_ylabel('t-SNE 2')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # True labels t-SNE plot
        colors_true = ['red' if label == 1 else 'blue' for label in y]
        axes[2].scatter(X_tsne[:, 0], X_tsne[:, 1], c=colors_true, alpha=0.7, s=50)
        axes[2].set_title('True Labels - t-SNE')
        axes[2].set_xlabel('t-SNE 1')
        axes[2].set_ylabel('t-SNE 2')
        
        # Create custom legend
        import matplotlib.patches as mpatches
        blue_patch = mpatches.Patch(color='blue', label='No Disease')
        red_patch = mpatches.Patch(color='red', label='Disease')
        axes[2].legend(handles=[blue_patch, red_patch])
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('clustering_tsne_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ t-SNE visualizations saved as 'clustering_tsne_analysis.png'")
    else:
        print(f"⚠️  Skipping t-SNE analysis (dataset too large: {X_scaled.shape[0]} samples)")
    
    # Cluster profiling - analyze feature means for each cluster
    print("\n📊 Creating Cluster Profiling...")
    
    # K-means cluster profiling
    kmeans_labels = kmeans_results[best_kmeans_k]['labels']
    kmeans_profiles = []
    
    for cluster_id in range(best_kmeans_k):
        cluster_mask = kmeans_labels == cluster_id
        cluster_data = X_scaled[cluster_mask]
        cluster_profile = {
            'cluster_id': cluster_id,
            'size': np.sum(cluster_mask),
            'mean_features': cluster_data.mean(axis=0).tolist(),
            'std_features': cluster_data.std(axis=0).tolist()
        }
        kmeans_profiles.append(cluster_profile)
        
        print(f"✅ K-Means Cluster {cluster_id}: {cluster_profile['size']} samples")
    
    # Hierarchical cluster profiling
    hierarchical_labels = hierarchical_results[best_hierarchical_n]['labels']
    hierarchical_profiles = []
    
    for cluster_id in range(best_hierarchical_n):
        cluster_mask = hierarchical_labels == cluster_id
        cluster_data = X_scaled[cluster_mask]
        cluster_profile = {
            'cluster_id': cluster_id,
            'size': np.sum(cluster_mask),
            'mean_features': cluster_data.mean(axis=0).tolist(),
            'std_features': cluster_data.std(axis=0).tolist()
        }
        hierarchical_profiles.append(cluster_profile)
        
        print(f"✅ Hierarchical Cluster {cluster_id}: {cluster_profile['size']} samples")
    
    return {
        'kmeans_profiles': kmeans_profiles,
        'hierarchical_profiles': hierarchical_profiles,
        'tsne_computed': X_scaled.shape[0] <= 1000
    }

def main():
    """
    Main function to execute complete clustering analysis
    """
    print("🚀 Starting Unsupervised Learning - Clustering Analysis")
    print("=" * 70)
    
    # 1. Load dataset
    data, dataset_source = load_dataset_for_clustering()
    if data is None:
        return
    
    # 2. Prepare clustering data
    X, y, feature_names, needs_scaling = prepare_clustering_data(data)
    
    # 3. Scale features if needed
    X_scaled, scaler = scale_features_for_clustering(X, needs_scaling)
    
    # 4. Elbow method analysis for optimal K
    optimal_k_info = elbow_method_analysis(X_scaled, max_k=10)
    
    # 5. Perform K-means clustering
    kmeans_results, best_kmeans_k = perform_kmeans_clustering(X_scaled, optimal_k_info)
    
    # 6. Perform hierarchical clustering
    hierarchical_results, best_hierarchical_n, linkage_matrices = perform_hierarchical_clustering(
        X_scaled, n_clusters_range=[2, 3, 4, 5, 6])
    
    # 7. Compare clusters with actual labels
    comparison_results = compare_clusters_with_labels(
        kmeans_results, hierarchical_results, y, best_kmeans_k, best_hierarchical_n)
    
    # 8. Create comprehensive visualizations
    create_clustering_visualizations(
        X_scaled, optimal_k_info, kmeans_results, hierarchical_results,
        linkage_matrices, comparison_results, y, best_kmeans_k, best_hierarchical_n)
    
    # 9. Create additional analysis
    additional_analysis = create_additional_analysis(
        X_scaled, kmeans_results, hierarchical_results, y, best_kmeans_k, best_hierarchical_n)
    
    # 10. Save all results
    save_clustering_results(
        kmeans_results, hierarchical_results, optimal_k_info,
        comparison_results, best_kmeans_k, best_hierarchical_n, dataset_source)
    
    print(f"\n🎉 Clustering Analysis Complete!")
    print("=" * 70)
    print("📋 Deliverables Created:")
    print("   ✔ K-Means clustering models with elbow method analysis")
    print("   ✔ Hierarchical clustering with dendrogram analysis")
    print("   ✔ Comprehensive clustering visualizations")
    print("   ✔ Cluster comparison with actual disease labels")
    print("   ✔ Performance metrics and model evaluation")
    print("   ✔ t-SNE visualization (if dataset size permits)")
    print("   ✔ Cluster profiling and analysis")
    
    # Summary of best results
    print(f"\n🏆 Best Clustering Results:")
    print(f"📊 K-Means (K={best_kmeans_k}):")
    print(f"   Silhouette Score: {kmeans_results[best_kmeans_k]['silhouette_score']:.3f}")
    print(f"   Adjusted Rand Index: {comparison_results['kmeans']['adjusted_rand_score']:.3f}")
    print(f"   Normalized Mutual Info: {comparison_results['kmeans']['normalized_mutual_info']:.3f}")
    
    print(f"\n🌳 Hierarchical ({best_hierarchical_n} clusters):")
    print(f"   Silhouette Score: {hierarchical_results[best_hierarchical_n]['silhouette_score']:.3f}")
    print(f"   Adjusted Rand Index: {comparison_results['hierarchical']['adjusted_rand_score']:.3f}")
    print(f"   Normalized Mutual Info: {comparison_results['hierarchical']['normalized_mutual_info']:.3f}")
    
    # Interpretation
    print(f"\n💡 Clustering Insights:")
    if comparison_results['kmeans']['adjusted_rand_score'] > 0.1:
        print("   ✅ K-Means clustering shows some alignment with disease labels")
    else:
        print("   ⚠️  K-Means clustering shows limited alignment with disease labels")
        
    if comparison_results['hierarchical']['adjusted_rand_score'] > 0.1:
        print("   ✅ Hierarchical clustering shows some alignment with disease labels")
    else:
        print("   ⚠️  Hierarchical clustering shows limited alignment with disease labels")
    
    best_method = "K-Means" if (kmeans_results[best_kmeans_k]['silhouette_score'] > 
                               hierarchical_results[best_hierarchical_n]['silhouette_score']) else "Hierarchical"
    print(f"   🏆 Overall best clustering method: {best_method}")
    
    return {
        'kmeans_results': kmeans_results,
        'hierarchical_results': hierarchical_results,
        'comparison_results': comparison_results,
        'optimal_k_info': optimal_k_info,
        'additional_analysis': additional_analysis
    }

if __name__ == "__main__":
    # Execute the main analysis
    results = main()