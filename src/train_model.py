import pandas as pd
import os
import joblib

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score

from data_preprocessing import clean_data


# 1. Load raw data
df_raw = pd.read_csv('data/server_temperature_data.csv')

# 2. Clean and engineer features
df = clean_data(df_raw)

# 3. Define features
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

X = df[features]
y = df['CPU_Temperature']

# 4. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5. Define models
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=50, n_jobs=-1, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=50, random_state=42)
}

best_model = None
best_score = -1
best_model_name = ""

print("Training and comparing models...\n")

for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    score = r2_score(y_test, predictions)

    cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
    print(f"{name}")
    print(f"Test R2 Score: {score:.4f}")
    print(f"Cross-validation Mean R2: {cv_scores.mean():.4f}")
    print("-" * 40)

    if score > best_score:
        best_score = score
        best_model = model
        best_model_name = name

print(f"\nBest Model Selected: {best_model_name}")
print(f"Best R2 Score: {best_score:.4f}")

# 6. Save best model
if not os.path.exists('models'):
    os.makedirs('models')

joblib.dump(best_model, 'models/server_temp_model.pkl')

# 7. Save test set
X_test_copy = X_test.copy()
X_test_copy['CPU_Temperature'] = y_test
X_test_copy.to_csv('data/test_set.csv', index=False)

print("\nModel saved to models/server_temp_model.pkl")
print("Test set saved to data/test_set.csv")
