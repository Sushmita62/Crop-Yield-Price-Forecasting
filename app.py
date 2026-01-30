import streamlit as st
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# Page config
st.set_page_config(
    page_title="Crop Yield Forecasting",
    page_icon="🌾",
    layout="wide"
)

# Title
st.title("🌾 Agricultural Yield & Price Forecasting System")
st.markdown("**Predict crop yields and market prices using ML & satellite data**")

# Sidebar
st.sidebar.header("📊 Model Information")
st.sidebar.markdown("""
**Models Available:**
- XGBoost (R² = 0.72)
- LSTM (R² = 0.63)
- Prophet (R² = 0.85)

**Data Sources:**
- MODIS NDVI (250m)
- ERA5 Climate Data
- CHIRPS Rainfall
""")

# Load model
@st.cache_resource
def load_model():
    return joblib.load('models/xgboost_model.pkl')

model = load_model()

# Main content
tab1, tab2, tab3 = st.tabs(["🎯 Prediction", "📈 Results", "ℹ️ About"])

with tab1:
    st.header("Crop Yield Prediction")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        ndvi = st.slider("NDVI (Vegetation Index)", 0.0, 1.0, 0.65, 0.01)
        rainfall = st.slider("Weekly Rainfall (mm)", 0, 500, 150, 10)
        temperature = st.slider("Temperature (°C)", 10, 45, 28, 1)
    
    with col2:
        humidity = st.slider("Humidity (%)", 20, 100, 75, 5)
        solar = st.slider("Solar Radiation (MJ/m²)", 5, 30, 18, 1)
        season = st.selectbox("Season", ["Kharif", "Rabi", "Summer"])
    
    with col3:
        st.metric("District", "Patna")
        st.metric("Year", "2024")
        st.metric("Week", "26")
    
    if st.button("🚀 Predict Yield", type="primary"):
        # Create feature vector (simplified - add all 19 features)
        season_encoded = {"Kharif": 1, "Rabi": 2, "Summer": 3}[season]
        
        # For demo, using simplified features
        features = np.array([[
            ndvi, rainfall, temperature, humidity, solar,
            ndvi*0.95, ndvi*0.90, rainfall*0.9, temperature*0.98,
            ndvi, rainfall/3, temperature,
            rainfall*10, ndvi*temperature, ndvi*rainfall,
            season_encoded, 0.5, 0.866, 8.5
        ]])
        
        prediction = model.predict(features)[0]
        
        st.success(f"### Predicted Yield: **{prediction:.0f} kg/hectare**")
        
        # Visualization
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = prediction,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Crop Yield (kg/ha)"},
            delta = {'reference': 2000},
            gauge = {
                'axis': {'range': [None, 3500]},
                'bar': {'color': "darkgreen"},
                'steps': [
                    {'range': [0, 1500], 'color': "lightgray"},
                    {'range': [1500, 2500], 'color': "gray"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 2800
                }
            }
        ))
        
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.header("Model Performance")
    
    results_df = pd.DataFrame({
        'Model': ['XGBoost', 'LSTM', 'Prophet'],
        'RMSE': [245.18, 284.16, 125.50],
        'MAE': [195.09, 225.18, 98.30],
        'R² Score': [0.7247, 0.6302, 0.8520],
        'Task': ['Yield', 'Yield', 'Price']
    })
    
    st.dataframe(results_df, use_container_width=True)
    
    # Bar chart
    fig = go.Figure(data=[
        go.Bar(name='RMSE', x=results_df['Model'], y=results_df['RMSE']),
        go.Bar(name='MAE', x=results_df['Model'], y=results_df['MAE'])
    ])
    fig.update_layout(title="Model Error Comparison", barmode='group')
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.header("About This Project")
    st.markdown("""
    ### 🎯 Objective
    Predict crop yields and market prices using satellite imagery, climate data, and machine learning.
    
    ### 📊 Dataset
    - **Region:** 15 districts in Bihar, India
    - **Time Period:** 2013-2024 (12 years)
    - **Samples:** 4,080 weekly observations
    - **Features:** 19 engineered features
    
    ### 🛠️ Technology Stack
    - **ML Models:** XGBoost, LSTM, Prophet
    - **Data Sources:** MODIS, ERA5, CHIRPS
    - **Framework:** TensorFlow, scikit-learn
    
    ### 👨‍💻 Developer
    **Sushmita** | Data Scientist
    
    [GitHub Repository](https://github.com/Sushmita62/Crop-Yield-Price-Forecasting)
    """)

# Footer
st.markdown("---")
st.markdown("Made with ❤️ for sustainable agriculture | © 2024")
