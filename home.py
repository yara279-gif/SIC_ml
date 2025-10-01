import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

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
        
        # Show model info in sidebar
        st.sidebar.subheader("Model Information")
        if hasattr(model, 'n_features_in_'):
            st.sidebar.write(f"**Expected features:** {model.n_features_in_}")
        if hasattr(model, 'feature_names_in_'):
            st.sidebar.write(f"**Feature names:** {list(model.feature_names_in_)}")
        else:
            st.sidebar.write("**Expected features (from dataset):** 10")
            st.sidebar.write("**Features:** Temperature[C], Humidity[%], TVOC[ppb], eCO2[ppm], Raw H2, Raw Ethanol, Pressure[hPa], PM1.0, NC0.5")
            
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
    df = generate_sample_data()
    
    if page == "Real-time Detection":
        show_realtime_detection(model)
    elif page == "Data Analysis":
        show_data_analysis(df)
    elif page == "Model Info":
        show_model_info(model)

def show_realtime_detection(model):
    st.header("🔍 Real-time Smoke Detection")
    
    st.info("Based on your dataset, providing all 10 features for accurate prediction")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Sensor Input Parameters")
        
        col1a, col1b = st.columns(2)
        
        with col1a:
            temperature = st.slider("Temperature (°C)", -20.0, 60.0, 20.0, 0.1)
            humidity = st.slider("Humidity (%)", 10.0, 80.0, 50.0, 0.1)
            tvoc = st.slider("TVOC (ppb)", 0, 60000, 0, 100)
            eco2 = st.slider("eCO2 (ppm)", 400, 60000, 400, 100)
            nc05 = st.slider("NC0.5", 0.0, 500.0, 0.0, 1.0)
        
        with col1b:
            raw_h2 = st.slider("Raw H2", 12000, 22000, 15000, 100)
            raw_ethanol = st.slider("Raw Ethanol", 15000, 25000, 19000, 100)
            pressure = st.slider("Pressure (hPa)", 930.0, 940.0, 935.0, 0.1)
            pm1 = st.slider("PM1.0", 0.0, 1000.0, 0.0, 1.0)
    
    with col2:
        st.subheader("Current Readings")
        st.metric("Temperature", f"{temperature}°C")
        st.metric("Humidity", f"{humidity}%")
        st.metric("TVOC", f"{tvoc} ppb")
        st.metric("eCO2", f"{eco2} ppm")
        st.metric("NC0.5", f"{nc05}")
    
    st.markdown("---")
    st.subheader("Fire Risk Assessment")
    
    if model is not None:
        if st.button("🔍 Analyze Fire Risk", type="primary", use_container_width=True):
            # Prepare input data with ALL 10 features in correct order
            input_data = np.array([[
                temperature,      # Temperature[C]
                humidity,         # Humidity[%]
                tvoc,             # TVOC[ppb]
                eco2,             # eCO2[ppm]
                raw_h2,           # Raw H2
                raw_ethanol,      # Raw Ethanol
                pressure,         # Pressure[hPa]
                pm1,              # PM1.0
                nc05              # NC0.5 - This was the missing feature!
            ]])
            
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
                
                # Show feature values used
                st.subheader("Input Features Used:")
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
                
                cols = st.columns(3)
                for idx, (feature, value) in enumerate(feature_values.items()):
                    with cols[idx % 3]:
                        st.metric(feature, value)
                    
            except Exception as e:
                st.error(f"Prediction error: {str(e)}")
                st.info("Make sure your model was trained with the same 10 features as shown above")
    else:
        st.warning("Model not loaded. Please check if the model file exists.")

def show_data_analysis(df):
    st.header("📊 Data Analysis")
    
    # Dataset overview
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Records", len(df))
    with col2:
        st.metric("Fire Alarms", df['Fire Alarm'].sum())
    with col3:
        st.metric("Alarm Rate", f"{df['Fire Alarm'].mean():.2%}")
    with col4:
        st.metric("Features", len(df.columns) - 1)  # Exclude target
    
    st.subheader("Feature Distributions")
    
    # Feature selection for analysis
    feature = st.selectbox("Select feature to analyze", 
                          ['Temperature[C]', 'Humidity[%]', 'TVOC[ppb]', 
                           'eCO2[ppm]', 'Raw H2', 'Raw Ethanol', 'Pressure[hPa]', 
                           'PM1.0', 'NC0.5'])
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Histogram
        fig, ax = plt.subplots(figsize=(10, 6))
        df[feature].hist(bins=30, ax=ax, alpha=0.7, color='skyblue')
        ax.set_title(f'Distribution of {feature}')
        ax.set_xlabel(feature)
        ax.set_ylabel('Frequency')
        st.pyplot(fig)
    
    with col2:
        # Box plot by fire alarm
        fig, ax = plt.subplots(figsize=(10, 6))
        df.boxplot(column=feature, by='Fire Alarm', ax=ax)
        ax.set_title(f'{feature} by Fire Alarm Status')
        plt.suptitle('')  # Remove automatic title
        st.pyplot(fig)
    
    # Correlation with target
    st.subheader("Feature Correlation with Fire Alarm")
    correlations = df.corr()['Fire Alarm'].drop('Fire Alarm').sort_values(ascending=False)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    correlations.plot(kind='bar', ax=ax, color='lightcoral')
    ax.set_title('Feature Correlation with Fire Alarm')
    ax.set_ylabel('Correlation Coefficient')
    plt.xticks(rotation=45)
    st.pyplot(fig)

def show_model_info(model):
    st.header("🤖 Model Information")
    
    if model is not None:
        st.success("Model loaded successfully!")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Model Details")
            st.write(f"**Model type:** {type(model).__name__}")
            
            if hasattr(model, 'n_features_in_'):
                st.write(f"**Number of features expected:** {model.n_features_in_}")
            else:
                st.write("**Number of features expected:** 10 (from dataset)")
                
            if hasattr(model, 'feature_names_in_'):
                st.write(f"**Feature names:** {list(model.feature_names_in_)}")
            else:
                st.write("""
                **Expected features (from your dataset):**
                - Temperature[C]
                - Humidity[%]
                - TVOC[ppb]
                - eCO2[ppm]
                - Raw H2
                - Raw Ethanol
                - Pressure[hPa]
                - PM1.0
                - NC0.5
                """)
        
        with col2:
            st.subheader("Model Parameters")
            try:
                params = model.get_params()
                # Show most important parameters
                important_params = {k: v for k, v in params.items() 
                                  if k in ['n_neighbors', 'weights', 'algorithm', 'leaf_size'] 
                                  or 'random_state' in k}
                for param, value in important_params.items():
                    st.write(f"**{param}:** {value}")
            except:
                st.write("Parameter information not available")
    else:
        st.error("No model loaded")

if __name__ == "__main__":
    main()