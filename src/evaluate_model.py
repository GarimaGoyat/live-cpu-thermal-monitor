import joblib
import os
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

model_path = 'models/server_temp_model.pkl'

if not os.path.exists(model_path):
    print("Model not found. Run train_model.py first.")
    exit()

# Load trained model
model = joblib.load(model_path)

# Load test data
if not os.path.exists('data/test_set.csv'):
    print("Test set not found. Run train_model.py first.")
    exit()

df_test = pd.read_csv('data/test_set.csv')

# IMPORTANT: Same feature list used during training
features = [
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
]

X = df_test[features]
y_actual = df_test['CPU_Temperature']

# Make predictions
y_pred = model.predict(X)

# Metrics
mae = mean_absolute_error(y_actual, y_pred)
rmse = mean_squared_error(y_actual, y_pred, squared=False)
r2 = r2_score(y_actual, y_pred)

print("-" * 40)
print("MODEL EVALUATION REPORT")
print("-" * 40)
print(f"Mean Absolute Error (MAE): {mae:.2f} °C")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f} °C")
print(f"R2 Score: {r2:.4f}")
print("-" * 40)

if r2 > 0.90:
    print("High Accuracy Model ✅")
else:
    print("Model may need improvement ⚠️")
