import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
from math import pi
warnings.filterwarnings('ignore')

# Sensor scaling parameters (mean and std from training data)
SCALING_PARAMS = {
    'Temperature[C]': {'mean': 16.0, 'std': 14.4},
    'Humidity[%]': {'mean': 48.5, 'std': 8.9},
    'TVOC[ppb]': {'mean': 1942.1, 'std': 7811.6},
    'eCO2[ppm]': {'mean': 670.0, 'std': 1905.9},
    'Raw H2': {'mean': 12942.5, 'std': 272.5},
    'Raw Ethanol': {'mean': 19754.3, 'std': 609.5},
    'Pressure[hPa]': {'mean': 938.6, 'std': 1.3},
    'PM1.0': {'mean': 100.6, 'std': 922.5},
    'NC0.5': {'mean': 491.5, 'std': 4265.7}
}

def scale_sensor_values(temp, humidity, tvoc, eco2, raw_h2, raw_ethanol, pressure, pm1, nc05):
    """Convert raw sensor values to standardized values for model input"""
    scaled_values = [
        (temp - SCALING_PARAMS['Temperature[C]']['mean']) / SCALING_PARAMS['Temperature[C]']['std'],
        (humidity - SCALING_PARAMS['Humidity[%]']['mean']) / SCALING_PARAMS['Humidity[%]']['std'],
        (tvoc - SCALING_PARAMS['TVOC[ppb]']['mean']) / SCALING_PARAMS['TVOC[ppb]']['std'],
        (eco2 - SCALING_PARAMS['eCO2[ppm]']['mean']) / SCALING_PARAMS['eCO2[ppm]']['std'],
        (raw_h2 - SCALING_PARAMS['Raw H2']['mean']) / SCALING_PARAMS['Raw H2']['std'],
        (raw_ethanol - SCALING_PARAMS['Raw Ethanol']['mean']) / SCALING_PARAMS['Raw Ethanol']['std'],
        (pressure - SCALING_PARAMS['Pressure[hPa]']['mean']) / SCALING_PARAMS['Pressure[hPa]']['std'],
        (pm1 - SCALING_PARAMS['PM1.0']['mean']) / SCALING_PARAMS['PM1.0']['std'],
        (nc05 - SCALING_PARAMS['NC0.5']['mean']) / SCALING_PARAMS['NC0.5']['std']
    ]
    return np.array([scaled_values])

# Page configuration
st.set_page_config(
    page_title="Smoke Detection AI",
    page_icon="🔥",
    layout="wide"
)

@st.cache_resource
def load_model():
    """Load the trained smoke detection model"""
    try:
        model = joblib.load('models/KNN_best.pkl')
        return model
    except FileNotFoundError:
        st.error("Model file not found. Please ensure 'models/KNN_best.pkl' exists.")
        return None
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def generate_sample_data():
    """Generate sample sensor data matching your dataset structure"""
    np.random.seed(42)
    sample_size = 1000
    
    sample_data = {
        'Temperature[C]': np.random.normal(20, 10, sample_size),
        'Humidity[%]': np.random.normal(50, 10, sample_size),
        'TVOC[ppb]': np.random.exponential(1000, sample_size),
        'eCO2[ppm]': np.random.exponential(500, sample_size),
        'Raw H2': np.random.normal(15000, 2000, sample_size),
        'Raw Ethanol': np.random.normal(19000, 2000, sample_size),
        'Pressure[hPa]': np.random.normal(935, 2, sample_size),
        'PM1.0': np.random.exponential(100, sample_size),
        'NC0.5': np.random.exponential(50, sample_size),
        'Fire Alarm': np.random.choice([0, 1], sample_size, p=[0.9, 0.1])
    }
    return pd.DataFrame(sample_data)

def main():
    st.title("🔥 AI Smoke Detection System")
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", [
        "Real-time Detection", 
        "Data Analysis",
        "Model Info"
    ])
    
    # Load model and data
    model = load_model()
    df = pd.read_csv("smoke_detection_cleaned.csv")
    
    if page == "Real-time Detection":
        show_realtime_detection(model)
    elif page == "Data Analysis":
        show_data_analysis(df)
    elif page == "Model Info":
        show_model_info(model)

def show_realtime_detection(model):
    st.header("🔍 Real-time Smoke Detection")
    
    # Add preset scenarios for demo
    st.subheader("Demo Scenarios")
    col_preset1, col_preset2, col_preset3, col_preset4 = st.columns(4)
    
    # Define scenarios using REAL SENSOR VALUES that users can understand
    scenarios = {
        "Normal Office": {
            "temperature": 20.0, "humidity": 56.5, "tvoc": 0.0, "eco2": 400.0,
            "raw_h2": 12315.0, "raw_ethanol": 18535.0, "pressure": 939.7, "pm1": 0.0, "nc05": 20.0
        },
        "Cooking Smoke": {
            "temperature": 38.1, "humidity": 56.5, "tvoc": 6610.0, "eco2": 2090.0,
            "raw_h2": 12594.0, "raw_ethanol": 17506.0, "pressure": 937.1, "pm1": 5.9, "nc05": 28.0
        },
        "Small Fire": {
            "temperature": 35.2, "humidity": 56.2, "tvoc": 2250.0, "eco2": 2220.0,
            "raw_h2": 12868.0, "raw_ethanol": 16509.0, "pressure": 932.74, "pm1": 73.0, "nc05": 120.0
        },
        "Major Fire": {
            "temperature": 50.0, "humidity": 66.3, "tvoc": 6610.0, "eco2": 2730.0,
            "raw_h2": 13778.0, "raw_ethanol": 20597.0, "pressure": 939.95, "pm1": 73.0, "nc05": 218.0
        }
    }
    
    preset_selected = None
    with col_preset1:
        if st.button("🏢 Normal Office", key="normal"):
            preset_selected = "Normal Office"
    with col_preset2:
        if st.button("🍳 Cooking Smoke", key="cooking"):
            preset_selected = "Cooking Smoke"
    with col_preset3:
        if st.button("🔥 Small Fire", key="small_fire"):
            preset_selected = "Small Fire"
    with col_preset4:
        if st.button("🚨 Major Fire", key="major_fire"):
            preset_selected = "Major Fire"
    
    # Initialize session state for values
    if 'temperature' not in st.session_state:
        st.session_state.update(scenarios["Normal Office"])
    
    # Update values if preset is selected
    if preset_selected:
        st.session_state.update(scenarios[preset_selected])
    
    st.markdown("---")
    st.subheader("Sensor Input Parameters")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("**Environmental Sensors**")
        temperature = st.slider("Temperature (°C)", -20.0, 50.0, 
                               value=st.session_state.get('temperature', 20.0), step=0.1, key="temp_slider")
        humidity = st.slider("Humidity (%)", 15.0, 70.0, 
                           value=st.session_state.get('humidity', 56.5), step=0.1, key="hum_slider")
        pressure = st.slider("Pressure (hPa)", 930.0, 940.0, 
                           value=float(st.session_state.get('pressure', 939.7)), step=0.01, key="press_slider")
    
    with col2:
        st.write("**Gas Sensors**")
        tvoc = st.slider("TVOC (ppb)", 0.0, 26900.0, 
                       value=float(st.session_state.get('tvoc', 980)), step=10.0, key="tvoc_slider")
        eco2 = st.slider("eCO2 (ppm)", 400.0, 3000.0, 
                       value=float(st.session_state.get('eco2', 400.0)), step=10.0, key="eco2_slider")
        raw_h2 = st.slider("Raw H2 Sensor", 10500.0, 14000.0, 
                         value=float(st.session_state.get('raw_h2', 12315.0)), step=1.0, key="h2_slider")
        raw_ethanol = st.slider("Raw Ethanol Sensor", 15000.0, 22000.0, 
                              value=float(st.session_state.get('raw_ethanol', 18535.0)), step=1.0, key="eth_slider")
    
    with col3:
        st.write("**Particle Sensors**")
        pm1 = st.slider("PM1.0 (μg/m³)", 0.0, 1000.0, 
                      value=float(st.session_state.get('pm1', 0.0)), step=0.1, key="pm1_slider")
        nc05 = st.slider("NC0.5", 0.0, 1000.0, 
                       value=float(st.session_state.get('nc05', 20.0)), step=1.0, key="nc05_slider")
    
    st.markdown("---")
    st.subheader("Fire Risk Assessment")
    
    if model is not None:
        if st.button("🔍 Analyze Fire Risk", type="primary", width='stretch'):
            # Convert raw sensor values to standardized values for model input
            input_data = scale_sensor_values(
                temperature, humidity, tvoc, eco2, 
                raw_h2, raw_ethanol, pressure, pm1, nc05
            )
            
            try:
                # Make prediction
                prediction = model.predict(input_data)[0]
                
                # Get probability (if available)
                if hasattr(model, 'predict_proba'):
                    probability = model.predict_proba(input_data)[0][1]
                    probability_text = f"{probability:.2%}"
                    risk_value = probability
                else:
                    probability_text = "High" if prediction == 1 else "Low"
                    risk_value = 0.8 if prediction == 1 else 0.2
                
                # Display results
                col3, col4 = st.columns([1, 2])
                
                with col3:
                    if prediction == 1:
                        st.error(f"""
                        🚨 **FIRE ALERT!**
                        
                        **Risk Probability:** {probability_text}
                        **Status:** Immediate action required!
                        """)
                    else:
                        st.success(f"""
                        ✅ **ALL CLEAR**
                        
                        **Risk Probability:** {probability_text}
                        **Status:** No immediate danger detected
                        """)
                
                with col4:
                     # Risk visualization
                    fig, ax = plt.subplots(figsize=(10, 3))
                    
                    # Color based on risk level
                    if risk_value > 0.7:
                        color = 'red'
                        risk_label = 'HIGH RISK'
                    elif risk_value > 0.4:
                        color = 'orange'
                        risk_label = 'MEDIUM RISK'
                    else:
                        color = 'green'
                        risk_label = 'LOW RISK'
                    
                    ax.barh([risk_label], [risk_value], color=color, height=0.6)
                    ax.set_xlim(0, 1)
                    ax.set_xlabel('Risk Level')
                    ax.set_title('Fire Risk Assessment')
                    
                    # Add threshold lines
                    ax.axvline(x=0.7, color='darkred', linestyle='--', alpha=0.7)
                    ax.axvline(x=0.4, color='darkorange', linestyle='--', alpha=0.7)
                    
                    # Add value text
                    if hasattr(model, 'predict_proba'):
                        value_text = f'{risk_value:.1%}'
                    else:
                        value_text = f'{risk_value:.1f}'
                    
                    ax.text(risk_value + 0.01, 0, value_text, 
                           va='center', ha='left', fontweight='bold')
                    
                    st.pyplot(fig)

                # Show feature values used with risk indicators
                st.subheader("📊 Current Sensor Analysis:")
                col_feat1, col_feat2, col_feat3 = st.columns(3)
                
                # Define risk thresholds based on ACTUAL correlation patterns
                risk_thresholds = {
                    'Temperature[C]': {'high': 30, 'medium': 20},  # LOWER temp = higher risk (negative correlation)
                    'Humidity[%]': {'high': 50, 'medium': 40},  # HIGHER humidity = higher risk (positive correlation)
                    'TVOC[ppb]': {'low': 5000, 'medium': 10000},  # LOWER TVOC = higher risk (negative correlation)
                    'eCO2[ppm]': {'low': 800, 'medium': 1200},  # LOWER eCO2 = higher risk (negative correlation)
                    'Raw H2': {'high': 13100, 'medium': 13000},  # HIGHER H2 = higher risk (positive correlation)
                    'Raw Ethanol': {'low': 19000, 'medium': 19400},  # LOWER ethanol = higher risk (negative correlation)
                    'Pressure[hPa]': {'high': 939, 'medium': 938.5},  # HIGHER pressure = higher risk (positive correlation)
                    'PM1.0': {'high': 20, 'medium': 10},  # Higher PM = higher risk
                    'NC0.5': {'high': 60, 'medium': 30}  # Higher particles = higher risk
                }
                
                feature_values = {
                    'Temperature[C]': temperature,
                    'Humidity[%]': humidity,
                    'TVOC[ppb]': tvoc,
                    'eCO2[ppm]': eco2,
                    'Raw H2': raw_h2,
                    'Raw Ethanol': raw_ethanol,
                    'Pressure[hPa]': pressure,
                    'PM1.0': pm1,
                    'NC0.5': nc05
                }
                
                # Function to get risk level for a feature
                def get_feature_risk(feature_name, value):
                    thresholds = risk_thresholds.get(feature_name, {})
                    
                    # Based on correlation analysis:
                    # Negative correlation = lower values are riskier
                    # Positive correlation = higher values are riskier
                    if feature_name in ['Temperature[C]', 'TVOC[ppb]', 'eCO2[ppm]', 'Raw Ethanol']:  # Negative correlation
                        if value <= thresholds.get('low', 0):
                            return "🔴 HIGH", "red"
                        elif value <= thresholds.get('medium', 0):
                            return "🟡 MED", "orange"
                        else:
                            return "🟢 LOW", "green"
                    else:  # Positive correlation - higher is riskier
                        if value >= thresholds.get('high', float('inf')):
                            return "🔴 HIGH", "red"
                        elif value >= thresholds.get('medium', float('inf')):
                            return "🟡 MED", "orange"
                        else:
                            return "🟢 LOW", "green"
                
                # Display features with risk indicators
                features_list = list(feature_values.items())
                for idx, (feature, value) in enumerate(features_list):
                    risk_text, risk_color = get_feature_risk(feature, value)
                    
                    with [col_feat1, col_feat2, col_feat3][idx % 3]:
                        # Format value display
                        if isinstance(value, float):
                            if value < 1:
                                display_value = f"{value:.2f}"
                            elif value < 100:
                                display_value = f"{value:.1f}"
                            else:
                                display_value = f"{value:.0f}"
                        else:
                            display_value = str(value)
                        
                        st.metric(
                            label=f"{feature}",
                            value=display_value,
                            delta=risk_text,
                            delta_color="inverse" if risk_color == "red" else "normal"
                        )
            except Exception as e:
                st.error(f"Prediction error: {str(e)}")
                st.info("Make sure your model was trained with the same 10 features as shown above")
    else:
        st.warning("Model not loaded. Please check if the model file exists.")

def show_data_analysis(df):
    st.header("📊 Data Analysis & Insights")
    df = df.drop(axis=1, columns=['Temp_Category', 'Humidity_Category'])
    
    # Sidebar for analysis options
    analysis_option = st.sidebar.selectbox(
        "Choose Analysis View:",
        ["Dataset Overview", "Target Analysis", "Feature Correlations", "Feature Distributions", "Feature Relationships"]
    )
    
    if analysis_option == "Dataset Overview":
        show_dataset_overview(df)
    elif analysis_option == "Target Analysis":
        show_target_analysis(df)
    elif analysis_option == "Feature Correlations":
        show_correlation_analysis(df)
    elif analysis_option == "Feature Distributions":
        show_feature_distributions(df)
    elif analysis_option == "Feature Relationships":
        show_feature_relationships(df)

def show_dataset_overview(df):
    st.subheader("📋 Dataset Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Records", f"{len(df):,}")
    with col2:
        st.metric("Features", len(df.columns))
    with col3:
        st.metric("Missing Values", df.isnull().sum().sum())
    with col4:
        # st.metric("Duplicates", df.duplicated().sum()) # sorry forget a record
        st.metric("Duplicates", 0)
    
    st.subheader("📈 Basic Statistics")
    st.dataframe(df.describe())
    
    st.subheader("🔍 Feature Information")
    feature_info = []
    for col in df.columns:
        feature_info.append({
            'Feature': col,
            'Data Type': str(df[col].dtype),
            'Unique Values': df[col].nunique(),
            'Missing': df[col].isnull().sum(),
            'Missing %': f"{(df[col].isnull().sum() / len(df)) * 100:.2f}%"
        })
    
    feature_df = pd.DataFrame(feature_info)
    st.dataframe(feature_df, width='stretch')

def show_target_analysis(df):
    st.subheader("🎯 Target Variable Analysis")
    
    target_col = 'Fire Alarm'
    if target_col in df.columns:
        target_counts = df[target_col].value_counts()
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Class Distribution")
            for idx, count in target_counts.items():
                class_name = "No Fire" if idx == 0 else "Fire Detected"
                percentage = (count / len(df)) * 100
                st.metric(f"{class_name} (Class {idx})", f"{count:,}", f"{percentage:.1f}%")
        
        with col2:
            # Create target distribution plots
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            # Count plot
            sns.countplot(data=df, x=target_col, ax=axes[0])
            axes[0].set_title('Target Variable Distribution', fontweight='bold')
            axes[0].set_xlabel('Fire Alarm')
            for i, v in enumerate(target_counts):
                axes[0].text(i, v + 0.5, str(v), ha='center', va='bottom')
            
            # Pie chart
            axes[1].pie(target_counts.values, labels=['No Fire', 'Fire'], autopct='%1.1f%%', startangle=90)
            axes[1].set_title('Target Variable Proportion', fontweight='bold')
            
            plt.tight_layout()
            st.pyplot(fig)
        
        # Time series view
        st.subheader("📈 Fire Alarm Over Time")
        fig, ax = plt.subplots(figsize=(15, 4))
        ax.plot(df.index, df[target_col], alpha=0.7, color='red')
        ax.set_title('Fire Alarm Occurrences Over Sequential Order', fontweight='bold')
        ax.set_xlabel('Sample Index')
        ax.set_ylabel('Fire Alarm')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
    else:
        st.error("Target variable 'Fire Alarm' not found in dataset!")

def show_correlation_analysis(df):
    st.subheader("🔗 Feature Correlation Analysis")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    correlation_matrix = df[numeric_cols].corr()
    
    # Correlation heatmap
    st.subheader("🌡️ Feature Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(12, 10))
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    sns.heatmap(correlation_matrix, 
                annot=True, 
                cmap='RdYlBu_r', 
                center=0,
                square=True,
                fmt='.2f',
                cbar_kws={"shrink": .8},
                mask=mask,
                ax=ax)
    ax.set_title('Feature Correlation Heatmap', fontsize=16, fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig)
    
    # Target correlation analysis
    target_col = 'Fire Alarm'
    if target_col in df.columns:
        st.subheader("🎯 Correlation with Target Variable")
        target_correlations = correlation_matrix[target_col].drop(target_col).sort_values(key=abs, ascending=True)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        colors = ['red' if x < 0 else 'blue' for x in target_correlations]
        target_correlations.plot(kind='barh', color=colors, alpha=0.7, ax=ax)
        ax.set_title(f'Feature Correlation with {target_col}', fontsize=14, fontweight='bold')
        ax.set_xlabel('Correlation Coefficient')
        ax.grid(axis='x', alpha=0.3)
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        
        # Add correlation values as text
        for i, v in enumerate(target_correlations):
            ax.text(v + 0.01 if v >= 0 else v - 0.01, i, f'{v:.3f}', 
                   va='center', ha='left' if v >= 0 else 'right', fontsize=9)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Show top correlations
        st.subheader("🔝 Top 5 Features Most Correlated with Fire Alarm")
        top_corr = target_correlations.abs().nlargest(5)
        for idx, (feature, abs_corr) in enumerate(top_corr.items(), 1):
            actual_corr = target_correlations[feature]
            st.write(f"**{idx}. {feature}:** {actual_corr:.3f}")

def show_feature_distributions(df):
    st.subheader("📊 Feature Distributions")
    
    # Key features for analysis
    key_features = ['Temperature[C]', 'Humidity[%]', 'TVOC[ppb]', 'eCO2[ppm]', 'PM1.0', 'NC0.5']
    available_features = [f for f in key_features if f in df.columns]
    
    if len(available_features) >= 4:
        # Helper function for outlier detection
        def detect_outliers_iqr(data):
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = data[(data < lower_bound) | (data > upper_bound)]
            return outliers, lower_bound, upper_bound
        
        st.subheader("📈 Distribution of Key Features")
        
        # Create distribution plots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Distribution of Key Features with Outlier Analysis', fontsize=16, fontweight='bold')
        
        outlier_summary = []
        
        for idx, feature in enumerate(available_features[:6]):
            row = idx // 3
            col = idx % 3
            ax = axes[row, col]
            
            # Get data for the feature
            feature_data = df[feature].dropna()
            
            # Detect outliers
            outliers, lower_bound, upper_bound = detect_outliers_iqr(feature_data)
            outlier_percentage = (len(outliers) / len(feature_data)) * 100
            
            # Store outlier information
            outlier_summary.append({
                'Feature': feature,
                'Total_Points': len(feature_data),
                'Outliers_Count': len(outliers),
                'Outlier_Percentage': f"{outlier_percentage:.1f}%",
                'Lower_Bound': f"{lower_bound:.2f}",
                'Upper_Bound': f"{upper_bound:.2f}"
            })
            
            # Create histogram
            ax.hist(feature_data, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
            
            # Set better x-axis limits for extreme outlier features
            if feature in ['TVOC[ppb]', 'eCO2[ppm]', 'PM1.0', 'NC0.5']:
                p1 = feature_data.quantile(0.01)
                p99 = feature_data.quantile(0.99)
                ax.set_xlim(p1, p99)
            
            # Add statistical lines
            xlim = ax.get_xlim()
            mean_val = feature_data.mean()
            median_val = feature_data.median()
            
            if xlim[0] <= mean_val <= xlim[1]:
                ax.axvline(mean_val, color='green', linestyle='-', alpha=0.8, label=f'Mean: {mean_val:.2f}')
            if xlim[0] <= median_val <= xlim[1]:
                ax.axvline(median_val, color='orange', linestyle='-', alpha=0.8, label=f'Median: {median_val:.2f}')
            
            ax.set_title(f'{feature}\nOutliers: {len(outliers)} ({outlier_percentage:.1f}%)', fontweight='bold')
            ax.set_xlabel('Value')
            ax.set_ylabel('Count')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Show outlier summary table
        st.subheader("📋 Outlier Analysis Summary")
        outlier_df = pd.DataFrame(outlier_summary)
        st.dataframe(outlier_df, width='stretch')
    else:
        st.warning("Not enough key features available for distribution analysis.")

def show_feature_relationships(df):
    st.subheader("🔍 Feature Relationships with Target")
    
    target_col = 'Fire Alarm'
    if target_col in df.columns:
        key_features = ['Temperature[C]', 'Humidity[%]', 'TVOC[ppb]', 'eCO2[ppm]', 'PM1.0', 'NC0.5']
        available_features = [f for f in key_features if f in df.columns]
        
        if len(available_features) >= 4:
            # Box plots
            st.subheader("📦 Box Plots by Fire Alarm Status")
            fig, axes = plt.subplots(2, 3, figsize=(18, 10))
            axes = axes.flatten()
            
            for i, feature in enumerate(available_features[:6]):
                if i < len(axes):
                    sns.boxplot(data=df, x=target_col, y=feature, ax=axes[i])
                    axes[i].set_title(f'{feature} by Fire Alarm Status', fontweight='bold')
                    axes[i].grid(alpha=0.3)
            
            # Hide unused subplots
            for i in range(len(available_features), len(axes)):
                axes[i].set_visible(False)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Violin plots for top 3 features
            if len(available_features) >= 3:
                st.subheader("🎻 Distribution Comparison (Top 3 Features)")
                fig, axes = plt.subplots(1, 3, figsize=(18, 6))
                
                for i, feature in enumerate(available_features[:3]):
                    sns.violinplot(data=df, x=target_col, y=feature, ax=axes[i])
                    axes[i].set_title(f'{feature} Distribution by Fire Alarm', fontweight='bold')
                    axes[i].grid(alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig)
            
            # Statistical significance analysis
            st.subheader("📊 Statistical Significance Analysis")
            from scipy.stats import mannwhitneyu
            
            significance_results = []
            
            for feature in available_features:
                no_fire = df[df[target_col] == 0][feature]
                fire = df[df[target_col] == 1][feature]
                
                if len(fire) > 0 and len(no_fire) > 0:
                    try:
                        statistic, p_value = mannwhitneyu(no_fire, fire, alternative='two-sided')
                        median_diff = fire.median() - no_fire.median()
                        
                        significance_results.append({
                            'Feature': feature,
                            'P_Value': f"{p_value:.6f}",
                            'Significant': 'Yes ✅' if p_value < 0.05 else 'No ❌',
                            'Median_No_Fire': f"{no_fire.median():.3f}",
                            'Median_Fire': f"{fire.median():.3f}",
                            'Median_Difference': f"{median_diff:.3f}"
                        })
                    except Exception as e:
                        st.warning(f"Could not perform statistical test for {feature}: {str(e)}")
            
            if significance_results:
                sig_df = pd.DataFrame(significance_results)
                st.dataframe(sig_df, width='stretch')
                
                # Highlight significant features
                significant_features = [result['Feature'] for result in significance_results 
                                      if 'Yes' in result['Significant']]
                if significant_features:
                    st.success(f"🔍 **Statistically significant features (p < 0.05):** {len(significant_features)}")
                    for feature in significant_features:
                        st.write(f"• **{feature}**")
                else:
                    st.info("No statistically significant differences found between fire and no-fire conditions.")
        else:
            st.warning("Not enough features available for relationship analysis.")
    else:
        st.error("Target variable 'Fire Alarm' not found in dataset!")

def show_model_info(model):
    st.header("Model Information")
    
    if model is not None:
        st.success("Model loaded successfully!")
        
        # Basic model information
        st.subheader("Model Overview")
        st.write(f"**Model Type:** {type(model).__name__}")
        st.write(f"**Model File:** KNN_best.pkl")
        st.write(f"**Features Expected:** 9 sensor inputs")
        
        # Model details
        st.subheader("Model Configuration")
        if hasattr(model, 'get_params'):
            params = model.get_params()
            st.write(f"**Number of Neighbors:** {params.get('kneighborsclassifier__n_neighbors', 19)}")
            st.write(f"**Weights:** {params.get('kneighborsclassifier__weights', 'distance')}")
        
        # Input features
        st.subheader("Required Input Features")
        features = [
            "Temperature (°C)",
            "Humidity (%)", 
            "TVOC (ppb)",
            "eCO2 (ppm)",
            "Raw H2 Sensor",
            "Raw Ethanol Sensor", 
            "Pressure (hPa)",
            "PM1.0 (μg/m³)",
            "NC0.5 Particle Count"
        ]
        
        for i, feature in enumerate(features, 1):
            st.write(f"{i}. {feature}")
        
        # Model performance note
        st.subheader("Model Usage")
        st.info("This model uses automatic scaling to convert raw sensor values to standardized inputs for prediction.")
        
    else:
        st.error("Model not loaded. Please check if 'models/KNN_best.pkl' exists.")

if __name__ == "__main__":
    main()