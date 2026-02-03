# Server CPU Temperature Prediction

## 📌 Problem Statement

Large-scale computing infrastructure such as cloud data centers face overheating risks due to high workloads and environmental conditions. Predicting server CPU temperature in advance can help in proactive cooling management and failure prevention.

This project builds a supervised machine learning regression model to estimate CPU temperature based on system workload and environmental parameters.

---

## 📊 Input Features

- CPU Utilization (%)
- Memory Usage (%)
- Clock Speed (GHz)
- Ambient Temperature (°C)
- Voltage (V)
- Current Load (A)

## 🎯 Target Variable

- CPU Temperature (°C)

---

## 🛠️ Tech Stack

- Python
- Pandas & NumPy
- Scikit-learn
- Matplotlib & Seaborn

---

## ⚙️ Project Workflow

1. Dataset creation and preprocessing
2. Feature scaling and data splitting
3. Model training using regression algorithms
4. Model evaluation using MAE, RMSE, and R² score
5. Temperature risk classification for proactive cooling decisions

---

## 📈 Machine Learning Models

- Linear Regression
- Random Forest Regressor

---

## 🚨 Proactive Cooling Logic

- **Safe**: Temperature < 70°C
- **Moderate Risk**: 70–80°C
- **High Risk**: > 80°C

---

## 📁 Project Structure

server-cpu-temperature-prediction/
│
├── data/
│   ├── server_temperature_data.csv
│   └── test_set.csv  # Saved test split used for evaluation (created by `src/train_model.py`)
│
├── src/
│   ├── train_model.py
│   ├── evaluate_model.py
│   └── predict.py
│
├── models/
│   └── server_temp_model.pkl  # Saved model (kept locally; not tracked in git)
│
├── docs/
│   ├── project_report.md
│   ├── results.md
│   └── model_accuracy_report.png  # Evaluation console screenshot
│
├── notebooks/
│   └── exploration.ipynb
│
├── data/cleaned_server_data.csv
├── README.md
└── requirements.txt

> Note: `models/server_temp_model.pkl` and `data/test_set.csv` are generated locally by running `python src/train_model.py`.

## 🔮 Future Scope

- Real-time server monitoring integration
- Deep learning-based temperature prediction
- Cloud deployment and API integration

## 🚀 How to Run the Prediction System

Since the trained model file (`.pkl`) exceeds GitHub's size limit, please follow these steps to run the AI on your local machine:

1. **Activate Environment**:
   `.\.venv\Scripts\Activate.ps1`
2. **Install Libraries**:
   `pip install -r requirements.txt`
3. **Train the Model (Do this once)**:
   Run `python src/train_model.py` to learn from the dataset and generate the local `server_temp_model.pkl` file and a held-out `data/test_set.csv` for evaluation.
4. **Evaluate the Model**:
   Run `python src/evaluate_model.py` to evaluate the saved model on the held-out `data/test_set.csv`. The script prints MAE and R² to the console and uses `data/test_set.csv` if available.
5. **Get Predictions**:
   Run `python src/predict.py` to see the AI forecast server temperatures.

Screenshot: **Evaluation output screenshot is included** at `docs/model_accuracy_report.png` (already added). See `docs/results.md` and `docs/project_report.md` for the model metrics and evaluation details.
