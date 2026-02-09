import joblib
import pandas as pd
import numpy as np

# 1. Load the model
model = joblib.load('models/server_temp_model.pkl')

# 2. Simulate a high-load server scenario with all 11 features
# This example shows a stressed server with high CPU and memory usage

print("="*60)
print("SCENARIO: High-Load Server (High Stress)")
print("="*60)

new_data = pd.DataFrame({
    'CPU_Usage': [90.0],              # 90% CPU utilization
    'CPU_Frequency': [3500.0],        # 3500 MHz
    'Memory_Usage': [85.0],           # 85% memory utilization
    'Disk_Usage': [70.0],             # 70% disk usage
    'Process_Count': [250],           # 250 running processes
    'Thread_Count': [2800],           # 2800 threads
    'GPU_Temperature': [55.0],        # 55°C GPU temp
    'Ambient_Temperature': [28.0],    # 28°C ambient (warm environment)
    'Voltage': [12.0],                # Standard 12V power supply
    'Current_Load': [9.0],            # Current load (CPU_Usage * 0.1)
    'Prev_CPU_Temperature': [65.0]    # Previous temperature was 65°C
})

print("\nInput Parameters:")
print(f"  CPU Usage: {new_data['CPU_Usage'][0]:.1f}%")
print(f"  Memory Usage: {new_data['Memory_Usage'][0]:.1f}%")
print(f"  GPU Temperature: {new_data['GPU_Temperature'][0]:.1f}°C")
print(f"  Ambient Temperature: {new_data['Ambient_Temperature'][0]:.1f}°C")
print(f"  Previous CPU Temp: {new_data['Prev_CPU_Temperature'][0]:.1f}°C")

# 3. Predict the Temperature
prediction = model.predict(new_data)[0]

print("\n" + "="*60)
print("PREDICTION RESULT")
print("="*60)
print(f"🔥 Predicted Server Temperature: {prediction:.2f}°C")

# 4. Risk assessment
print("\nRisk Assessment:")
if prediction < 70:
    print(f"  ✅ SAFE: Temperature {prediction:.2f}°C is in the safe zone (< 70°C)")
elif prediction < 80:
    print(f"  ⚠️  MODERATE RISK: Temperature {prediction:.2f}°C (70-80°C range)")
    print("     Consider activating cooling to prevent overheating")
else:
    print(f"  🔴 HIGH RISK: Temperature {prediction:.2f}°C (> 80°C)")
    print("     ALERT: High risk of overheating! Triggering proactive cooling.")

print("="*60)