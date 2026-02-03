import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib
import os
from data_preprocessing import clean_data

# 1. Load Data
df_raw = pd.read_csv('data/server_temperature_data.csv')
df = clean_data(df_raw)

# 2. Setup Features and Target
X = df[['CPU_Usage', 'CPU_Frequency', 'Memory_Usage', 'Process_Count', 'Thread_Count']]
y = df['CPU_Temperature']

# 3. Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train the AI
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5. Create the models folder if it doesn't exist
if not os.path.exists('models'):
    os.makedirs('models')

# 6. Save the model and the test set for evaluation
joblib.dump(model, 'models/server_temp_model.pkl')
X_test_copy = X_test.copy()
X_test_copy['CPU_Temperature'] = y_test
X_test_copy.to_csv('data/test_set.csv', index=False)
print("✅ SUCCESS: The model has been trained on the train split and saved to models/server_temp_model.pkl")
print("✅ Test set saved to data/test_set.csv")
