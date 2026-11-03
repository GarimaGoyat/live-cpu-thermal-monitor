# Server CPU Temperature Prediction

## 📌 Problem Statement

Large-scale computing infrastructure such as cloud data centers face overheating risks due to high workloads and environmental conditions. Predicting server CPU temperature in advance can help in proactive cooling management and failure prevention.

This project builds a supervised machine learning regression model to estimate CPU temperature based on system workload and environmental parameters.

---

## 🎯 Model Performance

**Achieved Results:**
- **R² Score**: 0.9528 (explains 95.28% of temperature variance) ✅
- **Mean Absolute Error (MAE)**: 1.81°C (excellent accuracy) ✅
- **Root Mean Squared Error (RMSE)**: 2.40°C (tight error bounds) ✅
- **Best Model**: Random Forest Regressor (auto-selected)
- **Status**: Dataset-validated prototype; real-world hardware accuracy is not yet established

The R², MAE, and RMSE above are held-out values from the supplied dataset. They must not be interpreted as real-world accuracy until predictions are validated against actual hardware temperature measurements.

---

## 📊 Input Features

**User-Provided Features:**
- CPU Utilization (%)
- Memory Usage (%)
- CPU Frequency (MHz)
- Disk Usage (%)
- Process Count
- Thread Count
- GPU Temperature (°C)
- Previous CPU Temperature (°C) - Captures thermal inertia

**Synthetic (Engineered) Features:**
- Ambient Temperature (°C) - Simulated with sinusoidal variation
- Voltage (V) - Constant 12V assumption
- Current Load (A) - Derived from CPU utilization (CPU_Usage × 0.1)

## 🎯 Target Variable

- CPU Temperature (°C)

---

## 🛠️ Tech Stack

- Python 3.12
- Pandas & NumPy (data processing)
- Scikit-learn (machine learning)
- Joblib (model serialization)
- Streamlit (interactive dashboard)
- Matplotlib & Seaborn (visualization)

---

## ⚙️ Project Workflow

1. **Data Preprocessing**: Raw data cleaning, handling missing values, and feature engineering
2. **Feature Engineering**: Synthetic parameter generation for environmental and thermal factors
3. **Model Training**: Multi-model training and automatic best model selection
4. **Model Evaluation**: Comparison using MAE, RMSE, R², and cross-validation
5. **Interactive Dashboard**: Real-time predictions with risk classification
6. **Temperature Risk Classification**: Proactive cooling decisions
7. **Live Windows Monitoring**: Measured resource telemetry and optional hardware temperatures

---

## 📈 Machine Learning Models

The system automatically trains and evaluates multiple models, selecting the best performer:

- **Linear Regression** - Baseline model
- **Random Forest Regressor** - Ensemble method with high accuracy
- **Gradient Boosting Regressor** - Advanced ensemble approach

**Best Model Selection**: Automatically selects the model with the highest R² score on the held-out test set, combined with cross-validation analysis for robustness.

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

### Step 1: Setup Environment
```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Step 2: Train the Model (Run Once)
```powershell
python src/train_model.py
```
**What this does:**
- Preprocesses raw data from `data/server_temperature_data.csv`
- Generates synthetic features (Ambient Temperature, Voltage, Current Load, Previous CPU Temperature)
- Trains and compares three models: Linear Regression, Random Forest, and Gradient Boosting
- **Automatically selects the best model** based on R² score and cross-validation analysis
- Saves the best model to `models/server_temp_model.pkl`
- Saves the held-out test set to `data/test_set.csv` for reproducible evaluation

### Step 3: Evaluate the Model
```powershell
python src/evaluate_model.py
```
**What this does:**
- Evaluates the saved model on the held-out test set (`data/test_set.csv`)
- Reports performance metrics: Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and R² Score
- Displays confidence assessment based on R² value

### Step 4: Run Interactive Dashboard
```powershell
streamlit run app.py
```
**Features:**
- **Interactive Input Controls**: Sliders and number inputs for all 11 features
- **Real-time Predictions**: Instant temperature predictions as you adjust parameters
- **Risk Classification**: Visual alerts for safe, moderate risk, and high-risk temperature zones
- **Live Monitoring**: Select `Live Monitoring` in the sidebar for 2–5 second Windows refreshes

### Live Windows Monitoring

Install and run [LibreHardwareMonitor](https://github.com/LibreHardwareMonitor/LibreHardwareMonitor) as an administrator, enable its WMI interface, and leave it running. The dashboard reads CPU and GPU temperature sensors through `root\\LibreHardwareMonitor`; unavailable sensors are shown as unavailable and are never fabricated.

The live collector uses `psutil` for CPU utilization, memory, current CPU frequency, disk capacity, process count, thread count, disk I/O rates, and network rates. Disk and network rates are display-only. The model's `Disk_Usage` remains disk capacity percentage to match the original dataset. Ambient temperature and voltage are explicitly entered in the sidebar because the original training pipeline synthesized them. Current load remains `CPU_Usage * 0.1`, matching training. Prediction begins after a GPU reading and a previous CPU-temperature sample are available.

To install the live-monitoring dependencies and launch the app:
```powershell
cd D:\ML_PROJECT\server-thermal-intelligence\live-cpu-thermal-monitor
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

Enable `Log samples to data/live_monitoring_log.csv` to append timestamps, measured features, predictions, actual temperatures, I/O rates, and risk labels. This file is separate from `data/server_temperature_data.csv`.

### Step 5: Command-Line Predictions (Optional)
```powershell
python src/predict.py
```
Test the model with a simulated high-load server scenario.

