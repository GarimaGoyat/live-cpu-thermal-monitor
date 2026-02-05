import streamlit as st
import joblib
import pandas as pd
import numpy as np

st.set_page_config(page_title="Server CPU Temperature Predictor", layout="centered")

st.title("Server CPU Temperature Prediction System")
st.write("Predict CPU temperature for proactive cooling management.")

# Load trained model
model = joblib.load("models/server_temp_model.pkl")

st.subheader("Input Server Parameters")

cpu_usage = st.slider("CPU Usage (%)", 0.0, 100.0, 50.0)
cpu_frequency = st.number_input("CPU Frequency (MHz)", 1000.0, 5000.0, 3500.0)
memory_usage = st.slider("Memory Usage (%)", 0.0, 100.0, 40.0)
disk_usage = st.slider("Disk Usage (%)", 0.0, 100.0, 50.0)
process_count = st.number_input("Process Count", 1, 1000, 200)
thread_count = st.number_input("Thread Count", 1, 5000, 2500)
gpu_temp = st.slider("GPU Temperature (°C)", 20.0, 100.0, 40.0)

# Simulated environmental assumptions
ambient_temp = 25
voltage = 12
current_load = cpu_usage * 0.1

# Previous temperature assumption (can be user input too)
prev_temp = st.slider("Previous CPU Temperature (°C)", 20.0, 100.0, 50.0)

# Create dataframe
input_data = pd.DataFrame([[
    cpu_usage,
    cpu_frequency,
    memory_usage,
    disk_usage,
    process_count,
    thread_count,
    gpu_temp,
    ambient_temp,
    voltage,
    current_load,
    prev_temp
]], columns=[
    'CPU_Usage',
    'CPU_Frequency',
    'Memory_Usage',
    'Disk_Usage',
    'Process_Count',
    'Thread_Count',
    'GPU_Temperature',
    'Ambient_Temperature',
    'Voltage',
    'Current_Load',
    'Prev_CPU_Temperature'
])

# Prediction
if st.button("Predict Temperature"):
    prediction = model.predict(input_data)[0]

    st.subheader("Prediction Result")
    st.success(f"Predicted CPU Temperature: {prediction:.2f} °C")

    if prediction > 80:
        st.error("⚠️ Overheating Risk Detected! Activate Cooling System.")
    else:
        st.info("Temperature within safe operating range.")
