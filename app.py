import streamlit as st
import pandas as pd
import numpy as np
import pickle

# --- 1. LOAD THE SAVED DATA ---
with open('saved_steps.pkl', 'rb') as file:
    data = pickle.load(file)

xgb_model = data["model"]
scaler = data["scaler"]
columns = data["columns"]

# --- 2. BUILD THE UI ---
st.title("Software Developer Salary Predictor")

with st.expander("⚠️ Read Before Using: Model Context & Limitations"):
    st.markdown("""
    **1. Currency is in USD:** The model was trained on global data and outputs in US Dollars. 
    
    **2. Dataset Bias (The Product vs. Service Divide):** This model is trained on the Stack Overflow Developer Survey. Developers who take this survey are generally highly engaged and often work at product-based companies or MNCs. Therefore, for markets like India, these predictions reflect the upper-percentile startup/FAANG ecosystem, not the median service-based IT sector (e.g., mass recruiters).
    
    **3. The IC Premium:** You may notice that a Senior "Standard Developer" (Specialist) sometimes out-earns a "Manager". In top-tier tech, elite Individual Contributors (ICs) and core architects frequently command higher compensation than middle management.
    """)


# Education Dictionary (Translates UI to Model Integers)
ed_map = {
    "Primary / Secondary School": 2,
    "Some College / Associate Degree": 3,
    "Bachelor's Degree": 4,
    "Master's Degree": 5,
    "Post-Grad / PhD": 6,
    "Other": 2
}

Country = st.selectbox(
    "Country", 
    ("India", "United States of America", "United Kingdom of Great Britain and Northern Ireland", "Germany", "Other")
)
EdLevel = st.selectbox("Education Level", list(ed_map.keys()))
YearsProCode = st.slider("Years of Professional Coding", min_value=0, max_value=50, value=1)
remote_status = st.radio("Work Environment", ("Remote", "In-person"))
role = st.selectbox("Current Role", ("Standard Developer", "Manager", "Student", "Academic"))
skills = st.radio(
    "Skillset Category", 
    ("Basic", "Specialist"), 
    help="Basic includes standard languages. Specialist includes high-paying niche languages."
)

# --- 3. TRIGGER THE PREDICTION ---
if st.button("Predict Salary"):
    # Build the blank template
    df_template = pd.DataFrame(data=np.zeros((1, len(columns))), columns=columns)
    
    # Map the variables
    df_template["YearsCodePro"] = YearsProCode
    df_template["EdLevel"] = ed_map[EdLevel]  # The dictionary mapping you just wrote
    df_template["Country_" + Country] = 1
    df_template["RemoteWork_" + remote_status] = 1
    df_template["is_" + skills] = 1
    
    if role != "Standard Developer":
        df_template["is_" + role] = 1

    # Scale the data and Predict
    scaled_data = scaler.transform(df_template)
    prediction = xgb_model.predict(scaled_data)
    
    # Output the result to the UI

    st.write(f"### Predicted Salary: ${prediction[0]:,.2f}")
