# 🚀 **Complete Usage Guide: Server CPU Temperature Prediction System**

---

## 📋 **Table of Contents**

1. [Quick Start](#quick-start)
2. [Detailed Step-by-Step Instructions](#detailed-step-by-step-instructions)
3. [Interactive Dashboard Guide](#interactive-dashboard-guide)
4. [Python Scripts Overview](#python-scripts-overview)
5. [Troubleshooting](#troubleshooting)
6. [Project Structure](#project-structure)

---

## ⚡ **Quick Start**

For experienced users who want to get up and running immediately:

```powershell
# 1. Activate environment
.\.venv\Scripts\Activate.ps1

# 2. Install dependencies (if not done)
pip install -r requirements.txt

# 3. Train model (creates server_temp_model.pkl)
python src/train_model.py

# 4. Launch interactive dashboard
streamlit run app.py
```

Then access the dashboard at: **`http://localhost:8501`**

---

## 📝 **Detailed Step-by-Step Instructions**

### **Step 1: Activate Virtual Environment**

```powershell
.\.venv\Scripts\Activate.ps1
```

**What it does**: 
- Activates the isolated Python environment with all packages

**Expected output**: 
```
(.venv) PS D:\AI Project\server-cpu-temperature-prediction>
```

**Why it matters**: 
- Keeps project dependencies isolated from your system Python
- Ensures all scripts use correct package versions

---

### **Step 2: Install Dependencies** (One-time only)

```powershell
pip install -r requirements.txt
```

**What it does**: 
- Installs all required packages listed in `requirements.txt`

**Required packages include**:
- `pandas` - Data manipulation
- `numpy` - Numerical computing
- `scikit-learn` - Machine learning models
- `joblib` - Model serialization
- `streamlit` - Interactive dashboard
- `seaborn` - Data visualization
- `matplotlib` - Plotting

**Expected output**: 
```
Successfully installed pandas-2.3.3 numpy-2.4.2 scikit-learn-1.8.0 ... (all packages)
```

**Notes**:
- Only run this once per environment
- Subsequent runs will skip already-installed packages

---

### **Step 3: Explore Data** (Optional)

```powershell
jupyter notebook notebooks/data_exploration.ipynb
```

**What it does**:
- Opens interactive Jupyter notebook
- Shows data loading and cleaning process
- Generates feature correlation heatmap
- Demonstrates all 11 features used in training

**Contains 4 cells**:
1. Load raw data & apply feature engineering
2. Verify data quality
3. Save cleaned dataset
4. Generate correlation heatmap

**Expected output**:
- Data shape and missing values summary
- Correlation matrix visualization
- List of all 11 features

---

### **Step 4: Train the Model** ⭐ **CRITICAL STEP**

```powershell
python src/train_model.py
```

**What it does**:
1. Loads raw data: `data/server_temperature_data.csv` (28,407 rows)
2. Preprocesses data:
   - Removes duplicates
   - Handles missing values
   - Generates 4 synthetic features
3. Trains 3 models in parallel:
   - Linear Regression
   - Random Forest Regressor
   - Gradient Boosting Regressor
4. Performs 5-fold cross-validation for each
5. **Automatically selects best model** based on R² score
6. Saves artifacts:
   - `models/server_temp_model.pkl` - Best trained model
   - `data/test_set.csv` - Held-out test set (20% of data)

**Duration**: 5-10 minutes

**Expected output**:
```
Training and comparing models...

Linear Regression
Test R2 Score: 0.9241
Cross-validation Mean R2: 0.8696
----------------------------------------
Random Forest
Test R2 Score: 0.9528 ✓ WINNER
Cross-validation Mean R2: 0.8697
----------------------------------------
Gradient Boosting
Test R2 Score: 0.9466
Cross-validation Mean R2: 0.8849
----------------------------------------

Best Model Selected: Random Forest
Best R2 Score: 0.9528

Model saved to models/server_temp_model.pkl
Test set saved to data/test_set.csv
```

**Key insights**:
- Random Forest selected as best performer (R² = 0.9528)
- Model explains **95.28%** of temperature variance
- Cross-validation scores confirm generalization

---

### **Step 5: Evaluate Model Performance** (Optional)

```powershell
python src/evaluate_model.py
```

**What it does**:
- Loads trained model from `models/server_temp_model.pkl`
- Evaluates on held-out test set: `data/test_set.csv`
- Calculates metrics: MAE, RMSE, R²
- Provides quality assessment

**Duration**: < 1 minute

**Expected output**:
```
----------------------------------------
MODEL EVALUATION REPORT
----------------------------------------
Mean Absolute Error (MAE): 1.81 °C
Root Mean Squared Error (RMSE): 2.40 °C
R2 Score: 0.9528
----------------------------------------
High Accuracy Model ✅
```

**Metrics explanation**:
- **MAE (1.81°C)**: Average prediction error
- **RMSE (2.40°C)**: Penalizes larger errors
- **R² (0.9528)**: 95.28% of variance explained

---

### **Step 6: Test Predictions** (Optional but Recommended)

```powershell
python src/predict.py
```

**What it does**:
- Loads trained model
- Creates a simulated high-load server scenario:
  - 90% CPU Usage
  - 85% Memory Usage
  - 55°C GPU Temperature
  - 28°C Ambient Temperature
- Makes temperature prediction
- Provides risk assessment

**Duration**: < 1 minute

**Expected output**:
```
============================================================
SCENARIO: High-Load Server (High Stress)
============================================================

Input Parameters:
  CPU Usage: 90.0%
  Memory Usage: 85.0%
  GPU Temperature: 55.0°C
  Ambient Temperature: 28.0°C
  Previous CPU Temp: 65.0°C

============================================================
PREDICTION RESULT
============================================================
🔥 Predicted Server Temperature: 67.81°C

Risk Assessment:
  ✅ SAFE: Temperature 67.81°C is in the safe zone (< 70°C)
============================================================
```

**Temperature Zones**:
- ✅ **Safe**: < 70°C (Low risk)
- ⚠️ **Moderate Risk**: 70-80°C (Alert required)
- 🔴 **High Risk**: > 80°C (Critical - activate cooling)

---

### **Step 7: Launch Interactive Dashboard** 🎨 **RECOMMENDED**

```powershell
streamlit run app.py
```

**What it does**:
- Launches web-based interactive interface
- Loads trained model for real-time predictions
- Provides user-friendly controls for all 11 features

**Duration**: Startup takes 10-15 seconds

**Expected output**:
```
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.x.x:8501
```

**Access dashboard**: Open browser to `http://localhost:8501`

---

## 🎨 **Interactive Dashboard Guide**

### **Dashboard Features**

The Streamlit dashboard provides an interactive interface to the model:

#### **Input Controls**

| Feature | Type | Range | Default | Purpose |
|---------|------|-------|---------|---------|
| CPU Usage | Slider | 0-100% | 50% | CPU utilization |
| CPU Frequency | Number Input | 1000-5000 MHz | 3500 | Processor speed |
| Memory Usage | Slider | 0-100% | 40% | RAM utilization |
| Disk Usage | Slider | 0-100% | 50% | Storage utilization |
| Process Count | Number Input | 1-1000 | 200 | Running processes |
| Thread Count | Number Input | 1-5000 | 2500 | Active threads |
| GPU Temperature | Slider | 20-100°C | 40° | GPU thermal state |
| Previous CPU Temp | Slider | 20-100°C | 50° | Previous reading (thermal inertia) |

#### **How to Use**

1. **Adjust Sliders**: Move sliders to change CPU/Memory/GPU values
2. **Enter Numbers**: Type precise values for Frequency/Process/Thread count
3. **Click "Predict Temperature"**: Triggers model prediction
4. **View Results**:
   - Predicted temperature with 2 decimal precision
   - Color-coded risk badge
   - Status message

#### **Interpretation**

**Output Examples**:

```
✅ Predicted CPU Temperature: 45.32 °C
   Temperature within safe operating range.
```

```
⚠️ Predicted CPU Temperature: 75.18 °C
   [Orange info badge] - Moderate risk detected
```

```
🔴 Predicted CPU Temperature: 82.55 °C
   Overheating Risk Detected! Activate Cooling System.
```

---

## 🔧 **Python Scripts Overview**

### **`src/data_preprocessing.py`**
Handles all data cleaning and feature engineering.

**Functions**:
```python
clean_data(df) → pd.DataFrame
```

**Process**:
1. Remove duplicates
2. Drop rows with missing values
3. Engineer 4 synthetic features
4. Return cleaned dataset with 11 features

**Usage**:
```python
from data_preprocessing import clean_data
df_cleaned = clean_data(df_raw)
```

---

### **`src/train_model.py`**
Trains and selects the best model.

**Key steps**:
1. Load and preprocess raw data
2. Train 3 models with cross-validation
3. Select model with highest R² score
4. Save model and test set

**Outputs**:
- `models/server_temp_model.pkl`
- `data/test_set.csv`

**Run**: `python src/train_model.py`

---

### **`src/evaluate_model.py`**
Evaluates trained model on test set.

**Metrics calculated**:
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score

**Run**: `python src/evaluate_model.py`

---

### **`src/predict.py`**
Makes predictions on custom scenarios.

**Features**:
- Loads trained model
- Creates example scenario
- Shows prediction with risk assessment

**Run**: `python src/predict.py`

---

### **`app.py`**
Streamlit web application for interactive predictions.

**Provides**:
- Web UI with sliders and inputs
- Real-time predictions
- Risk classification

**Run**: `streamlit run app.py`

---

### **`notebooks/data_exploration.ipynb`**
Jupyter notebook for exploratory data analysis.

**Cells**:
1. Data loading and feature engineering
2. Data quality verification
3. Save cleaned dataset
4. Generate correlation heatmap

**Run**: `jupyter notebook notebooks/data_exploration.ipynb`

---

## 🐛 **Troubleshooting**

### **Issue: "No module named pandas" or similar**

**Solution**:
```powershell
pip install -r requirements.txt
```

**Why**: Missing required packages

---

### **Issue: "Model not found" error**

**Solution**:
```powershell
python src/train_model.py
```

**Why**: Must train model before using it

---

### **Issue: ".venv/Scripts/Activate.ps1 cannot be loaded"**

**Solution** (Windows PowerShell):
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.\.venv\Scripts\Activate.ps1
```

**Why**: PowerShell security policy blocking script execution

---

### **Issue: Streamlit dashboard won't open**

**Solution**:
1. Check if port 8501 is available
2. Try alternate port:
```powershell
streamlit run app.py --server.port 8502
```

**Why**: Port 8501 might be in use by another app

---

### **Issue: "CSV file not found" in notebook**

**Solution**:
- Run from project root directory
- Ensure all data files exist in `data/` folder

**Why**: File paths are relative to working directory

---

## 📂 **Project Structure**

```
server-cpu-temperature-prediction/
│
├── data/
│   ├── server_temperature_data.csv       # Raw dataset (28,407 rows)
│   ├── cleaned_server_data.csv           # Processed dataset (28,079 rows, 11 features)
│   └── test_set.csv                      # Held-out test set (5,616 rows)
│
├── src/
│   ├── data_preprocessing.py             # Data cleaning & feature engineering
│   ├── train_model.py                    # Model training & selection
│   ├── evaluate_model.py                 # Model evaluation metrics
│   └── predict.py                        # Test predictions
│
├── models/
│   └── server_temp_model.pkl             # Trained Random Forest model
│
├── notebooks/
│   └── data_exploration.ipynb            # EDA and analysis
│
├── docs/
│   ├── project_report.md                 # Technical documentation
│   ├── results.md                        # Performance metrics
│   └── model_accuracy_report.png         # Evaluation visualization
│
├── app.py                                # Streamlit dashboard
├── README.md                             # Project overview
├── USAGE_GUIDE.md                        # This file
├── requirements.txt                      # Python dependencies
└── .gitignore                            # Git ignore rules
```

---

## 📊 **End-to-End Workflow Diagram**

```
Raw Data (28,407 rows, 9 features)
            ↓
    [Data Preprocessing]
    - Remove duplicates
    - Handle missing values
    - Engineer features (+4 synthetic)
            ↓
Cleaned Data (28,079 rows, 11 features)
            ↓
    [Train/Test Split: 80/20]
            ↓
    ┌─────────────────────────────────────┐
    │   Train 3 Models in Parallel:       │
    │   - Linear Regression               │
    │   - Random Forest ← BEST (R²=0.9528)│
    │   - Gradient Boosting               │
    └─────────────────────────────────────┘
            ↓
    [Save Best Model]
            ↓
[Evaluate on Test Set] → MAE: 1.81°C, RMSE: 2.40°C, R²: 0.9528
            ↓
    [Deploy for Use]
            ├── Command-line predictions (src/predict.py)
            └── Interactive dashboard (streamlit run app.py)
```

---

## ✅ **Validation Checklist**

Before considering the project ready for production:

- [ ] **Data**: All 11 features present and clean (28,079 rows)
- [ ] **Notebook**: All cells execute successfully without errors
- [ ] **Training**: Model trains and selects Random Forest (R² = 0.9528)
- [ ] **Evaluation**: Test metrics show MAE < 2°C, R² > 0.95
- [ ] **Predictions**: `predict.py` runs and outputs realistic values
- [ ] **Dashboard**: Streamlit app launches and accepts input
- [ ] **Documentation**: README, project_report, and results updated

---

## 📞 **Support & Next Steps**

### **For Quick Questions**:
- Check [Troubleshooting](#troubleshooting) section

### **For Detailed Information**:
- See [docs/project_report.md](docs/project_report.md) for technical details
- See [docs/results.md](docs/results.md) for performance metrics
- See [README.md](README.md) for project overview

### **To Extend the Project**:
1. Modify input ranges in `app.py` for dashboard
2. Retrain model with different parameters in `src/train_model.py`
3. Add new features in `src/data_preprocessing.py`

---

**Last Updated**: February 9, 2026
**Version**: 1.0
**Status**: ✅ Production Ready
