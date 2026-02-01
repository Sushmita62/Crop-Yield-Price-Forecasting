import streamlit as st
import joblib
import pandas as pd
import numpy as np

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="🌾 Crop Yield Forecasting",
    page_icon="🌾",
    layout="wide"
)

st.title("🌾 Agricultural Yield Forecasting System")
st.markdown("Predict crop yield using a trained **XGBoost model** based on satellite & climate features.")

# -------------------------------
# Load Model (cached)
# -------------------------------
@st.cache_resource
def load_model():
    return joblib.load("models/xgboost_model.pkl")

model = load_model()

st.success("✅ Model loaded successfully (XGBoost)")

# -------------------------------
# Sidebar Inputs
# -------------------------------
st.sidebar.header("📥 Input Parameters")

ndvi = st.sidebar.slider("NDVI (Vegetation Index)", 0.0, 1.0, 0.65, 0.01)
rainfall = st.sidebar.slider("Weekly Rainfall (mm)", 0, 500, 150, 5)
temperature = st.sidebar.slider("Temperature (°C)", 10, 45, 28, 1)
humidity = st.sidebar.slider("Relative Humidity (%)", 20, 100, 75, 1)
solar_rad = st.sidebar.slider("Solar Radiation (MJ/m²)", 5, 30, 18, 1)

season = st.sidebar.selectbox("Season", ["Kharif", "Rabi", "Summer"])
week = st.sidebar.slider("Week of Year", 1, 52, 26)

# -------------------------------
# Feature Engineering
# -------------------------------
season_map = {"Kharif": 1, "Rabi": 2, "Summer": 3}
season_encoded = season_map[season]

features = pd.DataFrame([{
    "ndvi": ndvi,
    "rainfall": rainfall,
    "temperature": temperature,
    "humidity": humidity,
    "solar_rad": solar_rad,

    "ndvi_lag1": ndvi,
    "ndvi_lag2": ndvi,
    "rainfall_lag1": rainfall,
    "temp_lag1": temperature,

    "ndvi_roll_mean": ndvi,
    "rain_roll_mean": rainfall,
    "temp_roll_mean": temperature,

    "cum_rainfall": rainfall * week,

    "ndvi_temp_interaction": ndvi * temperature,
    "ndvi_rain_interaction": ndvi * rainfall,

    "season_encoded": season_encoded,
    "week_sin": np.sin(2 * np.pi * week / 52),
    "week_cos": np.cos(2 * np.pi * week / 52),

    "vpd": 8.5  # assumed constant for demo
}])

# -------------------------------
# Prediction
# -------------------------------
st.markdown("---")
if st.button("🚀 Predict Crop Yield", use_container_width=True):
    prediction = model.predict(features)[0]

    st.subheader("🎯 Prediction Result")
    st.success(f"**Predicted Yield:** {prediction:.0f} kg/hectare")

    if prediction < 1500:
        st.warning("⚠️ Low Yield: Consider irrigation or crop management")
    elif prediction < 2500:
        st.info("ℹ️ Moderate Yield: Within expected range")
    else:
        st.success("✅ High Yield: Favorable growing conditions")

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.caption(
    "Built by **Sushmita Raj (IIT Kharagpur)** | "
    "Model: XGBoost | Data: MODIS, ERA5, CHIRPS"
)
