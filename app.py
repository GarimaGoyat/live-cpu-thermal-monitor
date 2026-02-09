import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# PAGE CONFIGURATION - DARK THEME + PROFESSIONAL LAYOUT
# ============================================================
st.set_page_config(
    page_title="System Monitoring - CPU Temperature Predictor",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "Professional Server CPU Temperature Prediction System\n\nPowered by Random Forest ML Model (R²=0.9528, MAE=1.81°C)"
    }
)

# Dark theme CSS
st.markdown("""
    <style>
    [data-testid="stAppViewContainer"] {
        background-color: #0e1117;
        color: #c9d1d9;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================
# SIDEBAR - MODEL PERFORMANCE METRICS
# ============================================================
with st.sidebar:
    st.markdown("### 📊 Model Performance")
    st.divider()
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="R² Score", value="95.28%", delta="Excellent")
    with col2:
        st.metric(label="MAE", value="1.81°C", delta="Accurate")
    
    st.markdown("""
    **Model Details:**
    - Algorithm: Random Forest Regressor
    - Training Samples: 28,079
    - Test R²: 0.9528
    - RMSE: 2.40°C
    
    **Risk Thresholds:**
    - 🟢 Safe: < 70°C
    - 🟡 Moderate: 65-80°C
    - 🔴 Critical: > 80°C
    """)
    
    st.divider()
    st.caption("Last Updated: Feb 10, 2026")

# ============================================================
# MAIN HEADER
# ============================================================
st.markdown("# 🖥️ System Monitoring Dashboard")
st.markdown("**Real-time CPU Temperature Prediction & Analysis**")
st.divider()

# Load trained model
model = joblib.load("models/server_temp_model.pkl")

# ============================================================
# INPUT SECTION - 2 COLUMN LAYOUT
# ============================================================
st.markdown("### ⚙️ Server Configuration")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Processor & Memory**")
    cpu_usage = st.slider("CPU Usage (%)", 0.0, 100.0, 50.0, key="cpu_usage")
    cpu_frequency = st.number_input("CPU Frequency (MHz)", 1000.0, 5000.0, 3500.0, 
                                    key="cpu_freq", help="Processor clock speed")
    memory_usage = st.slider("Memory Usage (%)", 0.0, 100.0, 40.0, key="memory")
    process_count = st.number_input("Process Count", 1, 1000, 200, key="process")

with col2:
    st.markdown("**Storage & Temperature**")
    disk_usage = st.slider("Disk Usage (%)", 0.0, 100.0, 50.0, key="disk")
    thread_count = st.number_input("Thread Count", 1, 5000, 2500, key="threads")
    gpu_temp = st.slider("GPU Temperature (°C)", 20.0, 100.0, 40.0, key="gpu_temp")
    prev_temp = st.slider("Previous CPU Temperature (°C)", 20.0, 100.0, 50.0, 
                         key="prev_temp", help="Thermal inertia from last reading")

# Simulated environmental assumptions
ambient_temp = 25.0
voltage = 12.0
current_load = cpu_usage * 0.1

# Create dataframe for prediction
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

# ============================================================
# PREDICTION BUTTON
# ============================================================
col_button = st.columns([2, 1, 2])
with col_button[1]:
    predict_btn = st.button("🔮 PREDICT TEMPERATURE", use_container_width=True)

if predict_btn:
    # Make prediction
    prediction = model.predict(input_data)[0]
    
    st.divider()
    st.markdown("### 📈 Prediction Results")
    
    # ============================================================
    # VISUAL GAUGE - COLOR CODED METRIC
    # ============================================================
    # Determine color and status
    if prediction < 70:
        status_color = "#2ecc71"  # Emerald Green
        status_text = "SAFE"
        status_emoji = "🟢"
    elif prediction < 80:
        status_color = "#f1c40f"  # Amber
        status_text = "MODERATE"
        status_emoji = "🟡"
    else:
        status_color = "#e74c3c"  # Crimson Red
        status_text = "CRITICAL"
        status_emoji = "🔴"
    
    # Display metrics
    metric_col1, metric_col2, metric_col3 = st.columns(3)
    
    with metric_col1:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, {status_color}20, {status_color}40); 
                    border: 2px solid {status_color}; border-radius: 10px; padding: 20px; text-align: center;">
            <h3 style="color: {status_color}; margin: 0;">{prediction:.2f}°C</h3>
            <p style="color: {status_color}; margin: 10px 0 0 0; font-weight: bold;">{status_emoji} {status_text}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with metric_col2:
        # Progress bar (20-100°C scale)
        temp_percentage = ((prediction - 20) / (100 - 20)) * 100
        st.progress(min(100, max(0, temp_percentage / 100)), text=f"Thermal Load: {temp_percentage:.1f}%")
    
    with metric_col3:
        st.metric("Temperature Range", f"20° - {prediction:.1f}° - 100°C", 
                 delta=f"{prediction - prev_temp:.1f}°C from previous")
    
    # ============================================================
    # DETAILED ALERTS & RECOMMENDATIONS
    # ============================================================
    st.markdown("---")
    
    if prediction < 70:
        st.success(f"""
        ✅ **System Status: OPTIMAL**
        
        Temperature **{prediction:.2f}°C** is within safe operating parameters.
        - No cooling intervention required
        - Current load is sustainable
        - System operating nominally
        """)
    elif prediction < 80:
        st.warning(f"""
        ⚠️ **System Status: ELEVATED**
        
        Temperature **{prediction:.2f}°C** indicates moderate thermal stress.
        - Monitor temperature trends closely
        - Consider adjusting workload distribution
        - Prepare cooling system if trend continues
        - Estimated safe time window: ~30 minutes
        """)
    else:
        st.error(f"""
        🔴 **System Status: CRITICAL**
        
        Temperature **{prediction:.2f}°C** has exceeded safe thresholds!
        - ACTIVATE COOLING SYSTEM IMMEDIATELY
        - Reduce workload urgently
        - Potential hardware thermal throttling detected
        - Risk of system shutdown in 5-10 minutes
        """)
    
    st.divider()
    
    # ============================================================
    # WHAT-IF ANALYSIS - 10 MINUTE FORECAST
    # ============================================================
    st.markdown("### 📊 10-Minute Thermal Forecast")
    
    # Simulate temperature trend based on current conditions
    time_minutes = np.array([0, 2, 4, 6, 8, 10])
    
    if cpu_usage > 80:
        # High load - temperature rises and stabilizes
        temperature_forecast = prediction + np.array([0, 1.5, 2.8, 3.5, 3.2, 2.9])
    elif cpu_usage > 50:
        # Medium load - slight temperature rise
        temperature_forecast = prediction + np.array([0, 0.5, 0.9, 1.1, 0.8, 0.5])
    else:
        # Low load - temperature stabilizes or decreases
        temperature_forecast = prediction + np.array([0, -0.3, -0.5, -0.6, -0.5, -0.3])
    
    # Create forecast dataframe
    forecast_df = pd.DataFrame({
        'Minutes': time_minutes,
        'Predicted Temperature (°C)': temperature_forecast
    })
    
    # Plot forecast
    fig, ax = plt.subplots(figsize=(12, 4))
    fig.patch.set_facecolor('#0e1117')
    ax.set_facecolor('#1c2128')
    
    ax.plot(forecast_df['Minutes'], forecast_df['Predicted Temperature (°C)'], 
           color='#58a6ff', linewidth=3, marker='o', markersize=8, label='Predicted Temperature')
    ax.axhline(y=70, color='#2ecc71', linestyle='--', linewidth=2, alpha=0.7, label='Safe Threshold (70°C)')
    ax.axhline(y=80, color='#e74c3c', linestyle='--', linewidth=2, alpha=0.7, label='Critical Threshold (80°C)')
    ax.fill_between(forecast_df['Minutes'], 0, 70, alpha=0.1, color='#2ecc71')
    ax.fill_between(forecast_df['Minutes'], 70, 80, alpha=0.1, color='#f1c40f')
    ax.fill_between(forecast_df['Minutes'], 80, 100, alpha=0.1, color='#e74c3c')
    
    ax.set_xlabel('Time (minutes)', color='#c9d1d9', fontsize=12)
    ax.set_ylabel('Temperature (°C)', color='#c9d1d9', fontsize=12)
    ax.set_title('Thermal Forecast - Next 10 Minutes', color='#c9d1d9', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.2, color='#30363d')
    ax.legend(loc='best', facecolor='#1c2128', edgecolor='#30363d', labelcolor='#c9d1d9')
    ax.tick_params(colors='#c9d1d9')
    
    st.pyplot(fig, use_container_width=True)
    
    # ============================================================
    # SYSTEM ANALYSIS METRICS
    # ============================================================
    st.markdown("---")
    st.markdown("### 🔍 System Analysis")
    
    analysis_col1, analysis_col2, analysis_col3, analysis_col4 = st.columns(4)
    
    with analysis_col1:
        st.metric("CPU Usage", f"{cpu_usage:.1f}%", 
                 delta="High" if cpu_usage > 70 else "Normal")
    
    with analysis_col2:
        st.metric("Memory Usage", f"{memory_usage:.1f}%", 
                 delta="High" if memory_usage > 70 else "Normal")
    
    with analysis_col3:
        st.metric("GPU Temperature", f"{gpu_temp:.1f}°C", 
                 delta=f"{gpu_temp - ambient_temp:.1f}°C above ambient")
    
    with analysis_col4:
        st.metric("Current Load", f"{current_load:.2f}A", 
                 delta="Derived from CPU usage")
    
    st.divider()
    st.caption("💡 **Tip:** Adjust input parameters above and click 'PREDICT TEMPERATURE' to explore different scenarios.")
