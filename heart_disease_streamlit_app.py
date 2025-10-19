"""
Heart Disease UCI Dataset - Data Preprocessing, EDA & PCA
Streamlit Web Application
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Heart Disease Analysis",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stAlert {
        padding: 1rem;
        margin: 1rem 0;
    }
    h1 {
        color: #FF4B4B;
        padding-bottom: 1rem;
    }
    h2 {
        color: #0068C9;
        padding-top: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'df' not in st.session_state:
    st.session_state.df = None
if 'df_processed' not in st.session_state:
    st.session_state.df_processed = None

# Feature definitions
CATEGORICAL_FEATURES = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
NUMERICAL_FEATURES = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']

# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

@st.cache_data
def load_heart_disease_data():
    """Load the Heart Disease UCI dataset"""
    try:
        from ucimlrepo import fetch_ucirepo
        heart_disease = fetch_ucirepo(id=45)
        X = heart_disease.data.features
        y = heart_disease.data.targets
        df = pd.concat([X, y], axis=1)
        return df, "ucimlrepo"
    except:
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
        column_names = [
            'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
            'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'num'
        ]
        df = pd.read_csv(url, names=column_names, na_values='?')
        return df, "url"

# ============================================================================
# PREPROCESSING FUNCTIONS
# ============================================================================

def handle_missing_values(df, strategy='median'):
    """Handle missing values"""
    df_cleaned = df.copy()
    
    if df_cleaned.isnull().sum().sum() == 0:
        return df_cleaned
    
    if strategy == 'drop':
        df_cleaned = df_cleaned.dropna()
    elif strategy == 'knn':
        imputer = KNNImputer(n_neighbors=5)
        features_to_impute = df_cleaned.select_dtypes(include=[np.number]).columns
        df_cleaned[features_to_impute] = imputer.fit_transform(df_cleaned[features_to_impute])
    else:
        num_cols = [col for col in NUMERICAL_FEATURES if col in df_cleaned.columns]
        if num_cols:
            num_imputer = SimpleImputer(strategy=strategy)
            df_cleaned[num_cols] = num_imputer.fit_transform(df_cleaned[num_cols])
        
        cat_cols = [col for col in CATEGORICAL_FEATURES if col in df_cleaned.columns]
        if cat_cols:
            cat_imputer = SimpleImputer(strategy='most_frequent')
            df_cleaned[cat_cols] = cat_imputer.fit_transform(df_cleaned[cat_cols])
    
    return df_cleaned

def encode_categorical_features(df):
    """Encode categorical features"""
    df_encoded = df.copy()
    
    if 'num' in df_encoded.columns:
        df_encoded['target'] = (df_encoded['num'] > 0).astype(int)
        df_encoded = df_encoded.drop('num', axis=1)
    
    categorical_to_encode = [col for col in CATEGORICAL_FEATURES if col in df_encoded.columns]
    
    if categorical_to_encode:
        df_encoded = pd.get_dummies(df_encoded, columns=categorical_to_encode, 
                                   prefix=categorical_to_encode, drop_first=True)
    
    return df_encoded

def scale_features(df, scaler_type='standard'):
    """Scale numerical features"""
    df_scaled = df.copy()
    
    numerical_cols = df_scaled.select_dtypes(include=[np.number]).columns
    if 'target' in numerical_cols:
        numerical_cols = numerical_cols.drop('target')
    
    numerical_cols = [col for col in numerical_cols if col in NUMERICAL_FEATURES]
    
    if len(numerical_cols) == 0:
        return df_scaled, None
    
    if scaler_type == 'minmax':
        scaler = MinMaxScaler()
    else:
        scaler = StandardScaler()
    
    df_scaled[numerical_cols] = scaler.fit_transform(df_scaled[numerical_cols])
    
    return df_scaled, scaler

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_target_distribution(df):
    """Plot target distribution"""
    if 'target' in df.columns:
        fig = make_subplots(rows=1, cols=2, specs=[[{'type':'pie'}, {'type':'bar'}]])
        
        target_counts = df['target'].value_counts()
        
        fig.add_trace(
            go.Pie(labels=['No Disease', 'Disease'], values=target_counts.values,
                  marker=dict(colors=['lightblue', 'lightcoral'])),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Bar(x=['No Disease', 'Disease'], y=target_counts.values,
                  marker=dict(color=['lightblue', 'lightcoral'])),
            row=1, col=2
        )
        
        fig.update_layout(height=400, title_text="Target Distribution")
        return fig
    return None

def plot_numerical_distributions(df):
    """Plot distributions of numerical features"""
    numerical_cols = [col for col in NUMERICAL_FEATURES if col in df.columns]
    
    if not numerical_cols:
        return None
    
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=[f'{col} Distribution' for col in numerical_cols[:6]]
    )
    
    for idx, col in enumerate(numerical_cols[:6]):
        row = idx // 3 + 1
        col_pos = idx % 3 + 1
        
        fig.add_trace(
            go.Histogram(x=df[col], name=col, marker=dict(color='skyblue')),
            row=row, col=col_pos
        )
    
    fig.update_layout(height=600, showlegend=False, title_text="Numerical Features Distribution")
    return fig

def plot_correlation_heatmap(df):
    """Plot correlation heatmap"""
    corr_features = [col for col in NUMERICAL_FEATURES if col in df.columns]
    if 'target' in df.columns:
        corr_features.append('target')
    
    if len(corr_features) > 1:
        corr_matrix = df[corr_features].corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu_r',
            zmid=0,
            text=corr_matrix.values,
            texttemplate='%{text:.2f}',
            textfont={"size": 10}
        ))
        
        fig.update_layout(
            title='Correlation Matrix',
            height=600,
            xaxis_title='Features',
            yaxis_title='Features'
        )
        return fig
    return None

def plot_boxplots_by_target(df):
    """Plot boxplots for numerical features by target"""
    if 'target' not in df.columns:
        return None
    
    numerical_cols = [col for col in NUMERICAL_FEATURES if col in df.columns]
    
    if not numerical_cols:
        return None
    
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=[f'{col} vs Heart Disease' for col in numerical_cols[:6]]
    )
    
    for idx, col in enumerate(numerical_cols[:6]):
        row = idx // 3 + 1
        col_pos = idx % 3 + 1
        
        for target_val in [0, 1]:
            data = df[df['target'] == target_val][col]
            name = 'No Disease' if target_val == 0 else 'Disease'
            
            fig.add_trace(
                go.Box(y=data, name=name, showlegend=(idx==0)),
                row=row, col=col_pos
            )
    
    fig.update_layout(height=600, title_text="Features vs Heart Disease")
    return fig

def plot_pca_variance(explained_variance, cumulative_variance):
    """Plot PCA variance analysis"""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=['Individual Component Variance (Scree Plot)', 'Cumulative Variance']
    )
    
    # Scree plot
    fig.add_trace(
        go.Scatter(x=list(range(1, len(explained_variance)+1)), 
                  y=explained_variance,
                  mode='lines+markers',
                  name='Individual Variance',
                  marker=dict(size=8, color='blue')),
        row=1, col=1
    )
    
    # Cumulative variance
    fig.add_trace(
        go.Scatter(x=list(range(1, len(cumulative_variance)+1)), 
                  y=cumulative_variance,
                  mode='lines+markers',
                  name='Cumulative Variance',
                  marker=dict(size=8, color='red')),
        row=1, col=2
    )
    
    # Add threshold lines
    for threshold in [0.80, 0.90, 0.95]:
        fig.add_hline(y=threshold, line_dash="dash", line_color="green", 
                     annotation_text=f"{threshold*100}%", row=1, col=2)
    
    fig.update_xaxes(title_text="Principal Component", row=1, col=1)
    fig.update_xaxes(title_text="Number of Components", row=1, col=2)
    fig.update_yaxes(title_text="Explained Variance Ratio", row=1, col=1)
    fig.update_yaxes(title_text="Cumulative Variance", row=1, col=2)
    
    fig.update_layout(height=400, showlegend=True)
    return fig

def plot_pca_2d_scatter(df_pca, variance_ratios):
    """Plot 2D PCA scatter"""
    if 'target' in df_pca.columns:
        fig = px.scatter(
            df_pca, x='PC1', y='PC2', color='target',
            labels={'PC1': f'PC1 ({variance_ratios[0]:.3f})', 
                   'PC2': f'PC2 ({variance_ratios[1]:.3f})'},
            color_continuous_scale='RdYlBu',
            title='PCA: First Two Components'
        )
    else:
        fig = px.scatter(
            df_pca, x='PC1', y='PC2',
            labels={'PC1': f'PC1 ({variance_ratios[0]:.3f})', 
                   'PC2': f'PC2 ({variance_ratios[1]:.3f})'},
            title='PCA: First Two Components'
        )
    
    fig.update_layout(height=500)
    return fig

def plot_pca_3d_scatter(df_pca, variance_ratios):
    """Plot 3D PCA scatter"""
    if 'target' in df_pca.columns:
        fig = px.scatter_3d(
            df_pca, x='PC1', y='PC2', z='PC3', color='target',
            labels={'PC1': f'PC1 ({variance_ratios[0]:.3f})', 
                   'PC2': f'PC2 ({variance_ratios[1]:.3f})',
                   'PC3': f'PC3 ({variance_ratios[2]:.3f})'},
            color_continuous_scale='RdYlBu',
            title='PCA: First Three Components (3D)'
        )
    else:
        fig = px.scatter_3d(
            df_pca, x='PC1', y='PC2', z='PC3',
            labels={'PC1': f'PC1 ({variance_ratios[0]:.3f})', 
                   'PC2': f'PC2 ({variance_ratios[1]:.3f})',
                   'PC3': f'PC3 ({variance_ratios[2]:.3f})'},
            title='PCA: First Three Components (3D)'
        )
    
    fig.update_layout(height=600)
    return fig

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    # Title and description
    st.title("❤️ Heart Disease UCI Dataset Analysis")
    st.markdown("""
    ### Data Preprocessing, EDA & Dimensionality Reduction (PCA)
    This application performs comprehensive data analysis on the Heart Disease UCI dataset.
    """)
    
    # Sidebar
    st.sidebar.title("⚙️ Configuration")
    st.sidebar.markdown("---")
    
    # Step selection
    analysis_step = st.sidebar.radio(
        "Select Analysis Step:",
        ["1️⃣ Data Loading", "2️⃣ Data Preprocessing", "3️⃣ EDA", "4️⃣ PCA Analysis"],
        index=0
    )
    
    st.sidebar.markdown("---")
    
    # ========================================================================
    # STEP 1: DATA LOADING
    # ========================================================================
    
    if analysis_step == "1️⃣ Data Loading":
        st.header("📊 Step 1: Data Loading")
        
        if st.button("🔄 Load Dataset", type="primary"):
            with st.spinner("Loading dataset..."):
                df, source = load_heart_disease_data()
                st.session_state.df = df
                st.session_state.data_loaded = True
                
                st.success(f"✅ Dataset loaded successfully from {source}!")
        
        if st.session_state.data_loaded:
            df = st.session_state.df
            
            # Display basic info
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Samples", df.shape[0])
            with col2:
                st.metric("Total Features", df.shape[1])
            with col3:
                st.metric("Missing Values", df.isnull().sum().sum())
            with col4:
                memory_mb = df.memory_usage(deep=True).sum() / 1024
                st.metric("Memory Usage", f"{memory_mb:.2f} KB")
            
            # Display dataset
            st.subheader("📋 Dataset Preview")
            st.dataframe(df.head(10), use_container_width=True)
            
            # Display info
            st.subheader("📈 Dataset Information")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Data Types:**")
                st.dataframe(pd.DataFrame({
                    'Column': df.columns,
                    'Type': df.dtypes.values,
                    'Non-Null': df.count().values,
                    'Unique': [df[col].nunique() for col in df.columns]
                }), use_container_width=True)
            
            with col2:
                st.write("**Missing Values:**")
                missing_df = pd.DataFrame({
                    'Column': df.columns,
                    'Missing': df.isnull().sum().values,
                    'Percentage': (df.isnull().sum() / len(df) * 100).values
                })
                missing_df = missing_df[missing_df['Missing'] > 0]
                if len(missing_df) > 0:
                    st.dataframe(missing_df, use_container_width=True)
                else:
                    st.info("No missing values found!")
            
            # Statistical summary
            st.subheader("📊 Statistical Summary")
            st.dataframe(df.describe(include='all'), use_container_width=True)
    
    # ========================================================================
    # STEP 2: PREPROCESSING
    # ========================================================================
    
    elif analysis_step == "2️⃣ Data Preprocessing":
        st.header("🔧 Step 2: Data Preprocessing")
        
        if not st.session_state.data_loaded:
            st.warning("⚠️ Please load the dataset first!")
            return
        
        df = st.session_state.df.copy()
        
        # Preprocessing options
        st.subheader("⚙️ Preprocessing Configuration")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            missing_strategy = st.selectbox(
                "Missing Value Strategy",
                ['median', 'mean', 'mode', 'knn', 'drop'],
                help="Choose how to handle missing values"
            )
        
        with col2:
            scaler_type = st.selectbox(
                "Feature Scaling Method",
                ['standard', 'minmax'],
                help="StandardScaler (mean=0, std=1) or MinMaxScaler (0-1 range)"
            )
        
        with col3:
            apply_encoding = st.checkbox("Apply One-Hot Encoding", value=True)
        
        if st.button("🚀 Run Preprocessing Pipeline", type="primary"):
            with st.spinner("Processing data..."):
                progress_bar = st.progress(0)
                
                # Step 1: Handle missing values
                st.info("🔄 Step 1: Handling missing values...")
                df_cleaned = handle_missing_values(df, strategy=missing_strategy)
                progress_bar.progress(33)
                
                # Step 2: Encode categorical features
                if apply_encoding:
                    st.info("🔄 Step 2: Encoding categorical features...")
                    df_encoded = encode_categorical_features(df_cleaned)
                else:
                    df_encoded = df_cleaned.copy()
                    if 'num' in df_encoded.columns:
                        df_encoded['target'] = (df_encoded['num'] > 0).astype(int)
                        df_encoded = df_encoded.drop('num', axis=1)
                progress_bar.progress(66)
                
                # Step 3: Scale features
                st.info("🔄 Step 3: Scaling numerical features...")
                df_scaled, scaler = scale_features(df_encoded, scaler_type=scaler_type)
                progress_bar.progress(100)
                
                st.session_state.df_processed = df_scaled
                st.success("✅ Preprocessing completed successfully!")
                
                # Display results
                st.subheader("📊 Preprocessing Results")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Original Shape", f"{df.shape}")
                    st.metric("Original Features", df.shape[1])
                    st.metric("Missing Values (Original)", df.isnull().sum().sum())
                
                with col2:
                    st.metric("Processed Shape", f"{df_scaled.shape}")
                    st.metric("Processed Features", df_scaled.shape[1])
                    st.metric("Missing Values (Processed)", df_scaled.isnull().sum().sum())
                
                # Show processed data
                st.subheader("📋 Processed Dataset Preview")
                st.dataframe(df_scaled.head(10), use_container_width=True)
                
                # Download options
                st.subheader("💾 Download Processed Data")
                col1, col2 = st.columns(2)
                
                with col1:
                    csv = df_scaled.to_csv(index=False)
                    st.download_button(
                        label="📥 Download as CSV",
                        data=csv,
                        file_name="heart_disease_processed.csv",
                        mime="text/csv"
                    )
                
                with col2:
                    st.download_button(
                        label="📥 Download Statistical Summary",
                        data=df_scaled.describe().to_csv(),
                        file_name="dataset_summary.csv",
                        mime="text/csv"
                    )
    
    # ========================================================================
    # STEP 3: EDA
    # ========================================================================
    
    elif analysis_step == "3️⃣ EDA":
        st.header("📊 Step 3: Exploratory Data Analysis")
        
        if not st.session_state.data_loaded:
            st.warning("⚠️ Please load the dataset first!")
            return
        
        df = st.session_state.df.copy()
        
        # EDA options
        eda_option = st.selectbox(
            "Select Visualization:",
            ["Target Distribution", "Numerical Features", "Correlation Analysis", 
             "Box Plots", "Feature Importance"]
        )
        
        if eda_option == "Target Distribution":
            st.subheader("🎯 Target Distribution")
            fig = plot_target_distribution(df if 'target' not in df.columns else st.session_state.df_processed)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
                
                if 'num' in df.columns or 'target' in (st.session_state.df_processed or {}).get('columns', []):
                    target_col = 'target' if st.session_state.df_processed is not None else 'num'
                    if target_col == 'num':
                        target_vals = (df[target_col] > 0).astype(int).value_counts()
                    else:
                        target_vals = st.session_state.df_processed[target_col].value_counts()
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("No Disease", target_vals[0])
                    with col2:
                        st.metric("Disease Present", target_vals[1])
        
        elif eda_option == "Numerical Features":
            st.subheader("📈 Numerical Features Distribution")
            fig = plot_numerical_distributions(df)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        
        elif eda_option == "Correlation Analysis":
            st.subheader("🔗 Correlation Matrix")
            fig = plot_correlation_heatmap(df)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        
        elif eda_option == "Box Plots":
            st.subheader("📦 Features vs Heart Disease")
            # Prepare data with binary target
            df_for_box = df.copy()
            if 'num' in df_for_box.columns:
                df_for_box['target'] = (df_for_box['num'] > 0).astype(int)
            
            fig = plot_boxplots_by_target(df_for_box)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        
        elif eda_option == "Feature Importance":
            st.subheader("🏆 Feature Importance (Correlation-based)")
            
            if st.session_state.df_processed is not None and 'target' in st.session_state.df_processed.columns:
                df_proc = st.session_state.df_processed
                numerical_cols = [col for col in df_proc.select_dtypes(include=[np.number]).columns 
                                if col != 'target']
                
                if numerical_cols:
                    correlations = df_proc[numerical_cols + ['target']].corr()['target'].drop('target')
                    correlations = correlations.abs().sort_values(ascending=False).head(10)
                    
                    fig = go.Figure(go.Bar(
                        x=correlations.values,
                        y=correlations.index,
                        orientation='h',
                        marker=dict(color=correlations.values, colorscale='Viridis')
                    ))
                    
                    fig.update_layout(
                        title='Top 10 Features by Correlation with Target',
                        xaxis_title='Absolute Correlation',
                        yaxis_title='Features',
                        height=500
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Please complete preprocessing first to see feature importance.")
    
    # ========================================================================
    # STEP 4: PCA ANALYSIS
    # ========================================================================
    
    elif analysis_step == "4️⃣ PCA Analysis":
        st.header("📐 Step 4: Principal Component Analysis (PCA)")
        
        if st.session_state.df_processed is None:
            st.warning("⚠️ Please complete preprocessing first!")
            return
        
        df_processed = st.session_state.df_processed
        
        # PCA Configuration
        st.subheader("⚙️ PCA Configuration")
        
        variance_threshold = st.slider(
            "Variance Retention Threshold",
            min_value=0.70,
            max_value=0.99,
            value=0.95,
            step=0.01,
            help="Percentage of variance to retain"
        )
        
        if st.button("🚀 Apply PCA", type="primary"):
            with st.spinner("Applying PCA..."):
                # Prepare data
                if 'target' in df_processed.columns:
                    X = df_processed.drop('target', axis=1)
                    y = df_processed['target']
                else:
                    X = df_processed
                    y = None
                
                # Apply PCA with all components first
                pca_full = PCA()
                pca_full.fit(X)
                
                explained_variance = pca_full.explained_variance_ratio_
                cumulative_variance = np.cumsum(explained_variance)
                
                # Determine optimal components
                n_components = np.argmax(cumulative_variance >= variance_threshold) + 1
                
                # Apply PCA with optimal components
                pca_optimal = PCA(n_components=n_components)
                X_pca = pca_optimal.fit_transform(X)
                
                # Create PCA DataFrame
                pca_columns = [f'PC{i+1}' for i in range(n_components)]
                df_pca = pd.DataFrame(X_pca, columns=pca_columns)
                
                if y is not None:
                    df_pca['target'] = y.values
                
                # Store in session state
                st.session_state.df_pca = df_pca
                st.session_state.pca_model = pca_optimal
                st.session_state.explained_variance = explained_variance
                st.session_state.cumulative_variance = cumulative_variance
                
                st.success(f"✅ PCA completed! Reduced from {X.shape[1]} to {n_components} components")
                
                # Display metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Original Features", X.shape[1])
                with col2:
                    st.metric("PCA Components", n_components)
                with col3:
                    st.metric("Variance Retained", f"{pca_optimal.explained_variance_ratio_.sum():.2%}")
                with col4:
                    reduction = ((X.shape[1] - n_components) / X.shape[1] * 100)
                    st.metric("Dimensionality Reduction", f"{reduction:.1f}%")
                
                # Variance plots
                st.subheader("📊 Variance Analysis")
                fig_variance = plot_pca_variance(explained_variance, cumulative_variance)
                st.plotly_chart(fig_variance, use_container_width=True)
                
                # Variance breakdown table
                st.subheader("📋 Component Variance Breakdown")
                variance_df = pd.DataFrame({
                    'Component': [f'PC{i+1}' for i in range(len(explained_variance))],
                    'Individual Variance': explained_variance,
                    'Cumulative Variance': cumulative_variance
                })
                st.dataframe(variance_df.head(10), use_container_width=True)
                
                # 2D Scatter Plot
                if n_components >= 2:
                    st.subheader("📈 PCA 2D Visualization")
                    fig_2d = plot_pca_2d_scatter(df_pca, pca_optimal.explained_variance_ratio_)
                    st.plotly_chart(fig_2d, use_container_width=True)
                
                # 3D Scatter Plot
                if n_components >= 3:
                    st.subheader("📈 PCA 3D Visualization")
                    fig_3d = plot_pca_3d_scatter(df_pca, pca_optimal.explained_variance_ratio_)
                    st.plotly_chart(fig_3d, use_container_width=True)
                
                # Component loadings
                st.subheader("🔍 Component Loadings")
                components_df = pd.DataFrame(
                    pca_optimal.components_.T,
                    columns=pca_columns,
                    index=X.columns
                )
                
                st.dataframe(components_df.style.background_gradient(cmap='RdBu_r', axis=None), 
                           use_container_width=True)
                
                # Top contributing features
                st.subheader("🏆 Top Contributing Features")
                for i in range(min(3, n_components)):
                    pc_name = f'PC{i+1}'
                    top_features = components_df[pc_name].abs().sort_values(ascending=False).head(5)
                    
                    with st.expander(f"{pc_name} - Top 5 Features (explains {pca_optimal.explained_variance_ratio_[i]:.2%} variance)"):
                        for feature, loading in top_features.items():
                            original_loading = components_df.loc[feature, pc_name]
                            st.write(f"**{feature}**: {original_loading:.3f} (|{loading:.3f}|)")
                
                # Download PCA data
                st.subheader("💾 Download PCA Results")
                col1, col2 = st.columns(2)
                
                with col1:
                    csv = df_pca.to_csv(index=False)
                    st.download_button(
                        label="📥 Download PCA Dataset",
                        data=csv,
                        file_name="heart_disease_pca.csv",
                        mime="text/csv"
                    )
                
                with col2:
                    loadings_csv = components_df.to_csv()
                    st.download_button(
                        label="📥 Download Component Loadings",
                        data=loadings_csv,
                        file_name="pca_loadings.csv",
                        mime="text/csv"
                    )
        
        # Show existing PCA results if available
        if hasattr(st.session_state, 'df_pca'):
            st.subheader("📋 Current PCA Dataset")
            st.dataframe(st.session_state.df_pca.head(10), use_container_width=True)
    
    # ========================================================================
    # FOOTER
    # ========================================================================
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    ### 📚 About
    This application performs:
    - ✅ Data Loading & Exploration
    - ✅ Data Preprocessing & Cleaning
    - ✅ Exploratory Data Analysis
    - ✅ PCA Dimensionality Reduction
    
    **Dataset**: Heart Disease UCI  
    **Source**: UCI Machine Learning Repository
    """)
    
    # Additional info in sidebar
    if st.session_state.data_loaded:
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 📊 Current Status")
        st.sidebar.success("✅ Data Loaded")
        
        if st.session_state.df_processed is not None:
            st.sidebar.success("✅ Data Preprocessed")
        else:
            st.sidebar.warning("⏳ Preprocessing Pending")
        
        if hasattr(st.session_state, 'df_pca'):
            st.sidebar.success("✅ PCA Applied")
        else:
            st.sidebar.warning("⏳ PCA Pending")

if __name__ == "__main__":
    main()
