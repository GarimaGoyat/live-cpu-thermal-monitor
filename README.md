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
- **Status**: Production Ready ✅

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
- **Environmental Simulation**: Included simulated ambient conditions for realistic scenarios

### Step 5: Command-Line Predictions (Optional)
```powershell
python src/predict.py
```
Test the model with a simulated high-load server scenario.

---

**Documentation References:**
- See [USAGE_GUIDE.md](USAGE_GUIDE.md) for **complete step-by-step instructions** on how to run everything ⭐
- See [docs/project_report.md](docs/project_report.md) for detailed technical implementation
- See [docs/results.md](docs/results.md) for complete model metrics and comparison analysis
- See [docs/model_accuracy_report.png](docs/model_accuracy_report.png) for evaluation visualization
