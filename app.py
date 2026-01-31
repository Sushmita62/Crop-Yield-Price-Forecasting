import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# Page config
st.set_page_config(
    page_title="Crop Yield Forecasting",
    page_icon="🌾",
    layout="wide"
)

# Title
st.title("🌾 Agricultural Yield & Price Forecasting System")
st.markdown("### Predict crop yields using ML & satellite data")

# Sidebar
with st.sidebar:
    st.header("📊 Project Info")
    st.markdown("""
    **Models Used:**
    - XGBoost (R² = 0.72) ⭐
    - LSTM (R² = 0.63)
    - Prophet (R² = 0.85)
    
    **Dataset:**
    - 15 districts in Bihar
    - 12 years (2013-2024)
    - 4,080 weekly samples
    
    **Data Sources:**
    - MODIS NDVI
    - ERA5 Climate
    - CHIRPS Rainfall
    """)
    
    st.markdown("---")
    st.markdown("**Developer:** Sushmita")
    st.markdown("[GitHub Repo](https://github.com/Sushmita62/Crop-Yield-Price-Forecasting)")

# Main tabs
tab1, tab2, tab3 = st.tabs(["🎯 Prediction Demo", "📈 Results", "ℹ️ About"])

with tab1:
    st.header("Interactive Yield Prediction")
    st.info("📝 **Note:** This is a demo using a simplified model. Full model requires trained .pkl files.")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Satellite Data")
        ndvi = st.slider("NDVI (Vegetation Index)", 0.0, 1.0, 0.65, 0.01, 
                        help="Higher values = healthier crops")
        
    with col2:
        st.subheader("Weather Data")
        rainfall = st.slider("Weekly Rainfall (mm)", 0, 500, 150, 10,
                           help="Total rainfall this week")
        temperature = st.slider("Temperature (°C)", 10, 45, 28, 1,
                              help="Average weekly temperature")
    
    with col3:
        st.subheader("Location & Time")
        district = st.selectbox("District", ["Patna", "Gaya", "Muzaffarpur", "Bhagalpur", "Darbhanga"])
        season = st.selectbox("Season", ["Kharif", "Rabi", "Summer"])
        week = st.slider("Week of Year", 1, 52, 26)
    
    st.markdown("---")
    
    if st.button("🚀 Predict Crop Yield", type="primary", use_container_width=True):
        # Simple prediction formula (demo only)
        base_yield = 2000
        ndvi_factor = (ndvi - 0.5) * 1000
        rain_factor = (rainfall - 100) * 2
        temp_factor = (28 - abs(temperature - 28)) * 10
        season_bonus = {"Kharif": 200, "Rabi": 100, "Summer": -100}[season]
        
        predicted_yield = base_yield + ndvi_factor + rain_factor + temp_factor + season_bonus
        predicted_yield = max(800, min(3500, predicted_yield))  # Clamp to realistic range
        
        # Display result
        col_a, col_b, col_c = st.columns([1, 2, 1])
        
        with col_b:
            st.success(f"### 🎯 Predicted Yield: **{predicted_yield:.0f} kg/hectare**")
            
            # Gauge chart
            fig = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = predicted_yield,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Crop Yield (kg/ha)", 'font': {'size': 24}},
                delta = {'reference': 2000, 'increasing': {'color': "green"}},
                gauge = {
                    'axis': {'range': [None, 3500], 'tickwidth': 1, 'tickcolor': "darkblue"},
                    'bar': {'color': "darkgreen"},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "gray",
                    'steps': [
                        {'range': [0, 1500], 'color': '#ffcccc'},
                        {'range': [1500, 2500], 'color': '#ffffcc'},
                        {'range': [2500, 3500], 'color': '#ccffcc'}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 2800
                    }
                }
            ))
            
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Interpretation
        if predicted_yield < 1500:
            st.warning("⚠️ **Low Yield Alert**: Consider irrigation or pest management")
        elif predicted_yield < 2500:
            st.info("ℹ️ **Moderate Yield**: Within normal range for Bihar")
        else:
            st.success("✅ **Excellent Yield**: Optimal growing conditions detected")

with tab2:
    st.header("Model Performance Metrics")
    
    # Results table
    results_df = pd.DataFrame({
        'Model': ['XGBoost 🏆', 'LSTM', 'Prophet'],
        'RMSE (kg/ha)': [245.18, 284.16, 125.50],
        'MAE (kg/ha)': [195.09, 225.18, 98.30],
        'R² Score': [0.7247, 0.6302, 0.8520],
        'Task': ['Yield Prediction', 'Yield Prediction', 'Price Forecasting']
    })
    
    st.dataframe(results_df, use_container_width=True, hide_index=True)
    
    # Comparison chart
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Error Metrics Comparison")
        fig1 = go.Figure(data=[
            go.Bar(name='RMSE', x=results_df['Model'], y=results_df['RMSE (kg/ha)'], marker_color='indianred'),
            go.Bar(name='MAE', x=results_df['Model'], y=results_df['MAE (kg/ha)'], marker_color='lightsalmon')
        ])
        fig1.update_layout(barmode='group', height=400)
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        st.subheader("R² Score (Higher is Better)")
        fig2 = go.Figure(data=[
            go.Bar(x=results_df['Model'], y=results_df['R² Score'], 
                   marker_color=['green', 'orange', 'darkgreen'])
        ])
        fig2.update_layout(height=400, yaxis_range=[0, 1])
        st.plotly_chart(fig2, use_container_width=True)
    
    # Feature Importance
    st.subheader("Top 10 Most Important Features")
    feature_importance = pd.DataFrame({
        'Feature': ['Season', 'Week (Temporal)', 'Temperature (4-week avg)', 'NDVI (4-week avg)', 
                   'Rainfall (4-week avg)', 'Cumulative Rainfall', 'VPD', 'NDVI × Temperature',
                   'NDVI (last week)', 'Rainfall (last week)'],
        'Importance (%)': [40.3, 24.7, 2.9, 2.9, 2.7, 2.5, 2.6, 2.5, 2.3, 2.2]
    })
    
    fig3 = go.Figure(go.Bar(
        x=feature_importance['Importance (%)'],
        y=feature_importance['Feature'],
        orientation='h',
        marker_color='lightgreen'
    ))
    fig3.update_layout(height=500, xaxis_title="Importance (%)", yaxis_title="Feature")
    st.plotly_chart(fig3, use_container_width=True)

with tab3:
    st.header("About This Project")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Objective")
        st.write("""
        Build an end-to-end ML pipeline to predict:
        - **Crop Yields** using satellite imagery and climate data
        - **Market Prices** using time series forecasting
        
        This helps farmers make data-driven decisions about:
        - When to plant and harvest
        - Expected yield for planning
        - Optimal selling time based on price forecasts
        """)
        
        st.subheader("📊 Dataset")
        st.write("""
        - **Region:** 15 districts in Bihar, India
        - **Time Period:** 2013-2024 (12 years)
        - **Granularity:** Weekly observations
        - **Total Samples:** 4,080 records
        - **Features:** 19 engineered features
        """)
        
    with col2:
        st.subheader("🛠️ Technology Stack")
        st.write("""
        **Data Sources:**
        - MODIS MOD13Q1 (NDVI - 250m resolution)
        - ERA5-Land (Temperature, Humidity)
        - CHIRPS (Rainfall - 5km resolution)
        
        **ML Models:**
        - XGBoost (Gradient Boosting)
        - LSTM (Deep Learning)
        - Prophet (Time Series)
        
        **Frameworks:**
        - TensorFlow 2.15
        - scikit-learn
        - Google Earth Engine
        """)
    
    st.markdown("---")
    
    st.subheader("📈 Key Findings")
    col_a, col_b, col_c = st.columns(3)
    
    with col_a:
        st.metric("Best Model", "XGBoost", "+15% vs LSTM")
    with col_b:
        st.metric("R² Score", "0.72", "72% accuracy")
    with col_c:
        st.metric("Training Samples", "4,080", "weekly data")
    
    st.markdown("---")
    
    st.subheader("🚀 Future Enhancements")
    st.write("""
    - Add real government yield records (vs synthetic)
    - Incorporate soil data (texture, pH, nutrients)
    - Expand to all 38 districts of Bihar
    - Deploy REST API for mobile app integration
    - Add SMS/WhatsApp alerts for farmers
    """)
    
    st.markdown("---")
    
    st.subheader("👨‍💻 Developer")
    st.write("""
    **Sushmita** | Data Scientist  
    📧 Contact: sushmita.raj@example.com  
    🔗 [GitHub](https://github.com/Sushmita62/Crop-Yield-Price-Forecasting)  
    💼 [LinkedIn](https://linkedin.com/in/yourprofile)
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Made with ❤️ for sustainable agriculture | © 2024</p>
    <p><a href='https://github.com/Sushmita62/Crop-Yield-Price-Forecasting'>⭐ Star this project on GitHub</a></p>
</div>
""", unsafe_allow_html=True)
```

---

## 🎯 FINAL STEPS

1. **Update `requirements.txt`** on GitHub (minimal version above)
2. **Update `app.py`** on GitHub (simplified version above)
3. **Wait 2-3 minutes** for Streamlit to rebuild
4. **App will work!** ✅

---

## 📱 YOUR LIVE APP URL

Once deployed successfully:
```
https://crop-yield-price-forecasting-cbpxkdenvthjcwdam7vhd2.streamlit.app
