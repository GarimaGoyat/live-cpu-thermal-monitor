import joblib
import os
import pandas as pd
import sys
from sklearn.metrics import mean_absolute_error, r2_score

model_path = 'models/server_temp_model.pkl'
if not os.path.exists(model_path):
    print(f"Error: model file '{model_path}' not found. Run: python src/train_model.py")
    sys.exit(1)

# 1. Load the trained model (and prefer a saved test set for evaluation)
model = joblib.load(model_path)

if os.path.exists('data/test_set.csv'):
    df_test = pd.read_csv('data/test_set.csv')
    X = df_test[['CPU_Usage', 'CPU_Frequency', 'Memory_Usage', 'Process_Count', 'Thread_Count']]
    y_actual = df_test['CPU_Temperature']
else:
    # fallback: create a fresh random split from cleaned data
    from sklearn.model_selection import train_test_split
    df = pd.read_csv('data/cleaned_server_data.csv')
    X_all = df[['CPU_Usage', 'CPU_Frequency', 'Memory_Usage', 'Process_Count', 'Thread_Count']]
    y_all = df['CPU_Temperature']
    X_train, X, y_train, y_actual = train_test_split(X_all, y_all, test_size=0.2, random_state=42)
    print("Warning: 'data/test_set.csv' not found — using a fresh random split for evaluation.")

# 3. Get predictions
y_pred = model.predict(X)

# 4. Calculate Accuracy Metrics
mae = mean_absolute_error(y_actual, y_pred)
r2 = r2_score(y_actual, y_pred)

print("-" * 30)
print("📊 MODEL ACCURACY REPORT")
print("-" * 30)
print(f"Mean Absolute Error (MAE): {mae:.2f}°C")
print(f"R2 Score (Accuracy): {r2:.4f}")
print("-" * 30)
if r2 > 0.80:
    print("Result: High Accuracy Model ✅")
else:
    print("Result: Model needs more tuning ⚠️")