# Crop-Yield-Price-Forecasting
This project focuses on predictive modeling of crop yield and market prices by integrating agro-climatic and remote sensing data. Using ML (Random Forest, LSTM) and time-series models (ARIMA, SARIMA, Prophet), it builds a scalable pipeline for climate-informed agricultural forecasting.
## Overview
This project aims to provide a reliable predictive model to aid in agricultural planning and market intelligence. We tackle a key challenge in the agricultural sector by leveraging real-world data to predict two critical variables:

**Crop Yield:** The total amount of a crop produced per unit area (in Tonnes/Hectare).

**Market Price:** The economic value of the crop (in Rupees/Quintal).

The pipeline is designed to be modular and scalable, allowing it to be adapted for different crops, regions, and time periods.

## Methodology
Our workflow follows a standard and rigorous data science process, ensuring a reproducible and verifiable project.

1. **Data Acquisition:** The project's foundation is a comprehensive, multi-source dataset for the entire state of Bihar from 2005 to 2023.

**Climate & Remote Sensing Data:** We use the Google Earth Engine (GEE) Python API to extract long-term time-series data, including rainfall, temperature, and the Normalized Difference Vegetation Index (NDVI), PET, solar radiation, humidity.

**Agricultural & Economic Data:** We integrate data from official government publications to capture ground-truth information on yield, price, and production cost.

2. **Data Preprocessing & Feature Engineering:** The raw data is cleaned, prepared, and merged into a single integrated dataset. This includes handling missing values and creating new variables like seasonal averages and lagged features.

3. **Predictive Modeling:** We train two separate models to address our dual forecasting goals.

**Yield Forecasting Model:** We use a Random Forest Regressor to predict crop yield based on climate and remote sensing data.

**Price Forecasting Model:** We use a Prophet or ARIMA time-series model to forecast market prices. The predicted yield from the first model is included as a key input feature for the price model.

4. **Evaluation & Visualization:** We evaluate our models using metrics like R-squared and Mean Absolute Error (MAE),  Root Mean Squared Error (RMSE), Mean Absolute Percentage Error  (MAPE) to ensure their reliability.

### Key Features
**End-to-End Pipeline:** Provides a complete solution from data collection to final predictions.

**Multi-Source Integration**: Seamlessly combines satellite, climate, and economic datasets.

**Advanced Modeling:** Implements both classical and modern machine learning models for robust forecasting(ML + Time-series + future DL).

**Scalable Architecture:** Designed to be easily adapted to different crops and regions.

### How to Run the Project
This project is designed to be run in a Python environment like Google Colab or a local Jupyter Notebook.

1. **Set up Environment:**
Use Python (Google Colab / Jupyter Notebook recommended).
Install all required Python libraries by running the following commands:

```Bash

!pip install pandas numpy scikit-learn matplotlib
!pip install earthengine-api
!pip install prophet # If you use the Prophet model
```
2. **Google Earth Engine Authentication:**
Authenticate your Google Earth Engine account to access the satellite data. Run this command and follow the instructions:

```Python

import ee
ee.Authenticate()
ee.Initialize()
```
3. **Run the Notebooks:**
The project code is structured into two main notebooks. Run the cells in each notebook in order:
```
notebooks/1_data_collection.ipynb

notebooks/2_modeling_and_forecasting.ipynb
```

### Repository Structure
```data/``` → Raw and processed datasets (```final_dataset.csv```).

```notebooks/``` → Jupyter/Colab notebooks containing the project code.

```results/ → Plots and model evaluation reports.

### Applications
This project demonstrates the potential of data-driven solutions in agriculture, with direct applications in:

Climate-Smart Agriculture: Guiding farmers on planting strategies based on climate forecasts.

Policy Planning: Aiding in the development of food security and market policies.

Market Intelligence: Providing businesses with actionable insights for supply chain management.
