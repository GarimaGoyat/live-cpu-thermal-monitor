import joblib
import pandas as pd

# 1. Load the model
model = joblib.load('models/server_temp_model.pkl')

# 2. Simulate a high-load server scenario
new_data = pd.DataFrame({
    'CPU_Usage': [90.0],
    'CPU_Frequency': [3.5],
    'Memory_Usage': [85.0],
    'Process_Count': [150],
    'Thread_Count': [1200]
})

# 3. Predict the Temperature
prediction = model.predict(new_data)
print(f"🔥 Predicted Server Temperature: {prediction[0]:.2f}°C")

if prediction[0] > 80:
    print("⚠️ ALERT: High risk of overheating! Triggering proactive cooling.")