# Tech Talent Market Analysis & Salary Predictor

**Live Application:** [https://predict-salaryy.streamlit.app/]

## Overview
This project is an end-to-end Machine Learning web application designed to predict software developer compensation based on global market data. It translates raw, messy survey data into a deployed, interactive tool that maps user inputs to complex backend constraints dynamically.

## Tech Stack
* **Data Engineering & Modeling:** Python, Pandas, NumPy, Scikit-Learn, XGBoost
* **Web Framework & Deployment:** Streamlit, Streamlit Community Cloud

## The Architecture

### 1. Data Pipeline
* **Source:** Stack Overflow Developer Survey.
* **Cleaning:** Dropped extreme outliers and missing values to prevent skewed distributions.
* **Feature Engineering:** Mapped complex string categories to standardized numerical values.
    * Applied **Ordinal Encoding** for education hierarchies (mapping degrees to integer weights).
    * Applied **One-Hot Encoding** for nominal categories (Country, Role, Remote Work).
* **Scaling:** Fitted the finalized dataset using Scikit-Learn scaling tools to normalize variance before model training.

### 2. Machine Learning Model
* Trained an **XGBoost Regressor** to handle the non-linear relationships between niche tech skills and compensation. 
* Pickled the trained model, the mathematical scaler, and the structural column arrays into a localized binary file for deployment extraction.

### 3. Application Backend
* Built a Streamlit interface that dynamically routes user UI selections into strict Pandas DataFrame constraints.
* Engineered a translation dictionary to map human-readable UI strings (e.g., "Master's Degree") into the exact Ordinal integer the loaded model expects.
* Processed single-row inference inputs through the pre-fitted Scikit-Learn transformer to prevent data leakage during real-time predictions.

## ⚠️ Market Context & Model Limitations (Read Before Using)

If you are testing this model for the Indian tech market, please note the following realities of the dataset:

1. **Output Currency:** Predictions are calculated in USD, not INR.
2. **Dataset Bias:** The model is trained on global Stack Overflow data. Developers who actively take this survey are highly engaged and typically work at well-funded product-based companies, startups, or MNCs. Therefore, predictions for the Indian market reflect the upper-percentile ecosystem (FAANG/Startups) rather than the median service-based IT sector (e.g., mass recruiters).
3. **The IC Premium:** You may notice that a Senior "Standard Developer" (Specialist) sometimes out-earns a "Manager". In top-tier tech hubs, elite Individual Contributors (ICs) frequently command higher compensation than middle management. The model accurately reflects this industry reality.
