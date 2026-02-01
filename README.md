# 🌾 Agricultural Yield & Price Forecasting System

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-R²%200.72-success.svg)](https://xgboost.ai/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 📊 Project Overview
End-to-end ML pipeline for crop yield prediction and market price forecasting using satellite imagery, climate data, and machine learning.
This repository demonstrates the **complete ML workflow**: data preparation, feature engineering, model training, evaluation, and inference-ready deployment code.

### **Key Results:**
- **XGBoost R²: 0.72** (Yield Prediction)
- **LSTM R²: 0.63** (Yield Prediction)  
- **Prophet R²: 0.85+** (Price Forecasting)
- **Dataset:** 15 districts, 12 years (2013-2024), 4,080 weekly samples

## 🛠️ Technologies Used
- **Remote Sensing:** Google Earth Engine (MODIS NDVI)
- **Climate Data:** ERA5-Land, CHIRPS
- **ML Models:** XGBoost, LSTM (TensorFlow), Prophet
- **Languages:** Python 3.10+
- **Key Libraries:** pandas, numpy, scikit-learn, xgboost, tensorflow, prophet

## 📁 Project Structure
```
├── data/
│   ├── ndvi_data.csv           # Satellite vegetation index
│   ├── weather_data.csv        # Climate variables
│   ├── market_price_data.csv   # Rice prices
│   ├── crop_yield_data.csv     # Yield targets
│   └── master_dataset.csv      # Merged dataset
├── models/
│   ├── xgboost_model.pkl       # Best yield model
│   ├── lstm_model.h5           # Deep learning model
│   └── prophet_model.pkl       # Price forecasting
├── notebooks/
│   └── Crop_Yield_Forecasting.ipynb
├── results/
│   ├── model_comparison.csv
│   └── feature_importance.png
├── requirements.txt
└── README.md
```

## 🚀 Quick Start

### Installation
```bash
# Clone repository
git clone https://github.com/Sushmita62/Crop-Yield-Price-Forecasting.git
cd Crop-Yield-Price-Forecasting

# Install dependencies
pip install -r requirements.txt
```

### Usage
```python
import joblib
import pandas as pd

# Load trained model
model = joblib.load('models/xgboost_model.pkl')

# Prepare input features
features = pd.DataFrame({
    'NDVI': [0.65],
    'rainfall_mm': [150],
    'temperature_c': [28],
    # ... other 16 features
})

# Predict yield
predicted_yield = model.predict(features)
print(f"Predicted Yield: {predicted_yield[0]:.0f} kg/ha")
```

## 📈 Model Performance

| Model | RMSE (kg/ha) | MAE (kg/ha) | R² Score | Use Case |
|-------|--------------|-------------|----------|----------|
| XGBoost | 245.18 | 195.09 | **0.7247** | Yield Prediction |
| LSTM | 284.16 | 225.18 | 0.6302 | Yield Prediction |
| Prophet | 125.50 | 98.30 | **0.8520** | Price Forecasting |

## 🔬 Methodology

### Data Sources
1. **NDVI (Vegetation Index):** MODIS MOD13Q1 (250m, 16-day composite)
2. **Weather Data:** 
   - Rainfall: CHIRPS (5km daily)
   - Temperature, Humidity: ERA5-Land (10km daily)
3. **Market Prices:** Synthetic data based on MSP and seasonal patterns
4. **Crop Yield:** Weather-correlated synthetic data

### Feature Engineering (19 features)
- Lagged features (1, 2, 4 weeks)
- Rolling statistics (4-week mean/std)
- Cumulative rainfall
- NDVI-weather interactions
- Cyclical time encoding
- Vapor Pressure Deficit (VPD)

### Model Architecture
**XGBoost:**
- n_estimators: 200
- max_depth: 6
- learning_rate: 0.05

**LSTM:**
- 2 LSTM layers (64, 32 units)
- Dropout: 0.2
- Dense layers: 16 → 1

## 📊 Results & Insights

### Key Findings
- **XGBoost outperforms LSTM** for tabular agricultural data
- **NDVI + rainfall** are strongest predictors (45% feature importance)
- **Seasonal patterns** critical for price forecasting
- **Weekly granularity** provides better temporal resolution than monthly

### Future Improvements
- Add soil data (texture, pH, nutrients)
- Incorporate real government yield records
- Expand to 38 districts (10× more data)
- Deploy REST API for real-time predictions
- Build interactive dashboard (Streamlit/Dash)

## 👨‍💻 Author
**Sushmita Raj**  
M.Tech Student, Indian Institute of Technology Kharagpur (2024–2026)

🔗 GitHub: https://github.com/Sushmita62  
🔗 LinkedIn: https://linkedin.com/in/sushmita-raj  

---
## 📄 License
This project is licensed under the **MIT License**.

---
## 🙏 Acknowledgments
- Google Earth Engine for satellite data
- Copernicus ERA5 for climate data
- CHIRPS Rainfall Dataset
- Bihar Department of Agriculture for domain insights
