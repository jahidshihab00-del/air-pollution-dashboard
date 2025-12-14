import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt
import pandas as pd

# Load model and scaler
model = joblib.load("air_pollution_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Air Pollution Health Risk", layout="centered")

st.title("🌫 Air Pollution Health Risk Prediction Dashboard")

st.markdown("This dashboard predicts health risk based on past air quality and weather conditions.")

st.header("🔢 Input Parameters")

temperature = st.slider("Temperature (°C)", 0, 50, 30)
humidity = st.slider("Humidity (%)", 0, 100, 60)
hour = st.slider("Hour of Day", 0, 23, 12)
day = st.selectbox("Day of Week", 
                   options=[0,1,2,3,4,5,6],
                   format_func=lambda x: ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][x])
pm_lag1 = st.slider("Previous PM2.5 (μg/m³)", 0, 500, 80)

# Prepare input
input_data = np.array([[temperature, humidity, hour, day, pm_lag1]])
input_scaled = scaler.transform(input_data)

# Predict
prediction = model.predict(input_scaled)[0]

risk_map = {
    0: "🟢 Good",
    1: "🟡 Moderate",
    2: "🔴 High Risk"
}

st.subheader("📊 Predicted Health Risk")
st.success(risk_map[prediction])

# Simple visualization
st.header("📈 Pollution Risk Indicator")

levels = ["Good", "Moderate", "High Risk"]
values = [1 if i == prediction else 0 for i in range(3)]

fig, ax = plt.subplots()
ax.bar(levels, values)
ax.set_ylabel("Risk Level")
st.pyplot(fig)
