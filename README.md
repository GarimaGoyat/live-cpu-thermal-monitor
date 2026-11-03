# Live CPU Thermal Monitor

This project predicts CPU temperature from workload, hardware, and thermal features, then presents the result in a Streamlit dashboard. It supports both manual what-if predictions and optional Windows live monitoring through `psutil` and LibreHardwareMonitor.

The repository includes model training, evaluation, command-line prediction, data analysis, and a live monitoring dashboard.

---

## Model Performance

The current live training pipeline compares Linear Regression, Random Forest, and Gradient Boosting, then selects the model with the highest held-out test $R^2$. The reproducible run in this repository selects **Random Forest**:

| Metric | Held-out test result |
| --- | ---: |
| $R^2$ | 0.9528 |
| MAE | 1.81 deg C |
| RMSE | 2.40 deg C |

These are dataset results, not validated hardware accuracy.

---

## 📊 Input Features

**User-Provided Features:**
- CPU Utilization (%)
- Memory Usage (%)
- CPU Frequency (MHz)
- Disk Usage (%)
- Process Count
- Thread Count
- GPU Temperature (deg C)
- Previous CPU Temperature (deg C) - Captures thermal inertia

**Synthetic (Engineered) Features:**
- Ambient Temperature (deg C) - Simulated with sinusoidal variation
- Voltage (V) - Constant 12V assumption
- Current Load (A) - Derived from CPU utilization (CPU_Usage x 0.1)

## Target Variable

- CPU Temperature (deg C)

---

## Tech Stack

- Python 3.10+
- Pandas & NumPy (data processing)
- Scikit-learn (machine learning)
- Joblib (model serialization)
- Streamlit (interactive dashboard)
- Matplotlib & Seaborn (visualization)

---

## What Is Included

1. `src/data_preprocessing.py` prepares the data and creates the model features.
2. `src/train_model.py` trains and compares regression models, then saves the best model.
3. `src/evaluate_model.py` reports MAE, RMSE, and $R^2$ on the saved test set.
4. `src/predict.py` runs a sample command-line prediction.
5. `app.py` provides manual predictions, risk levels, live Windows telemetry, charts, and optional CSV logging.
6. `src/eda.py`, `src/model_comparison.py`, `src/classification.py`, and `src/association_rules.py` provide additional analysis tools and saved result tables.

---

## 🚨 Proactive Cooling Logic

- **Safe**: Temperature < 70 deg C
- **Moderate Risk**: 70 to < 80 deg C
- **High Risk**: >= 80 deg C

---

## Project Structure

live-cpu-thermal-monitor/
├── app.py                              # Streamlit dashboard
├── data/                               # Source data and generated experiment outputs
├── models/server_temp_model.pkl        # Live dashboard model
├── models/engineered_model_experiment.pkl
├── src/                                # Training, evaluation, monitoring, and experiments
├── requirements.txt
└── README.md

`models/server_temp_model.pkl`, `models/engineered_model_experiment.pkl`, and several CSV files are generated artifacts. The live model is committed for convenience; Git LFS is recommended for future model versions because the model is approximately 94 MB.

## Limitations and Next Steps

- Ambient temperature and voltage are synthetic/operator-provided inputs in the live model.
- Live prediction requires a previous CPU temperature sample and a GPU temperature reading from LibreHardwareMonitor.
- Dataset performance does not establish accuracy on unseen hardware.
- Improve temporal validation, collect representative high-risk samples, and validate against calibrated hardware sensors before operational use.

## Run the Project

Use the following steps to run the project locally:

### 1. Create the environment
```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Train the live model
```powershell
python src/train_model.py
```
This preprocesses `data/server_temperature_data.csv`, compares three models by held-out test $R^2$, saves the selected model to `models/server_temp_model.pkl`, and writes `data/test_set.csv`.

### 3. Evaluate the live model
```powershell
python src/evaluate_model.py
```
This evaluates the saved model against `data/test_set.csv` and prints MAE, RMSE, and $R^2$.

### 4. Run the dashboard
```powershell
streamlit run app.py
```
The dashboard offers manual predictions for all 11 model inputs, three temperature risk zones, measured CPU/memory/disk/process/thread telemetry, optional CPU/GPU sensor readings, a rolling temperature chart, and optional CSV logging.

### Live Windows Monitoring

Install and run [LibreHardwareMonitor](https://github.com/LibreHardwareMonitor/LibreHardwareMonitor) as an administrator, enable its WMI interface, and leave it running. The dashboard reads CPU and GPU temperature sensors through `root\\LibreHardwareMonitor`; unavailable sensors are shown as unavailable and are never fabricated.

The live collector uses `psutil` for CPU utilization, memory, current CPU frequency, disk capacity, process count, thread count, disk I/O rates, and network rates. Disk and network rates are display-only. The model's `Disk_Usage` remains disk capacity percentage to match the original dataset. Ambient temperature and voltage are explicitly entered in the sidebar because the original training pipeline synthesized them. Current load remains `CPU_Usage * 0.1`, matching training. Prediction begins after a GPU reading and a previous CPU-temperature sample are available.

To install the live-monitoring dependencies and launch the app from this repository:
```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

Enable `Log samples to data/live_monitoring_log.csv` to append timestamps, measured features, predictions, actual temperatures, I/O rates, and risk labels. This file is separate from `data/server_temperature_data.csv`.

### 5. Command-line prediction (optional)
```powershell
python src/predict.py
```
This tests the saved model with a simulated high-load server scenario.

