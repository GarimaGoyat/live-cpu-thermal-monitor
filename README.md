# Live CPU Thermal Monitor

This project predicts CPU temperature from workload, hardware, and thermal features, then presents the result in a Streamlit dashboard. It supports both manual what-if predictions and optional Windows live monitoring through `psutil` and LibreHardwareMonitor.

The repository contains two related workflows:

- **Live prediction workflow**: `src/train_model.py`, `src/evaluate_model.py`, `src/predict.py`, and `app.py`. This workflow uses the 11-feature model saved as `models/server_temp_model.pkl`.
- **ML laboratory workflow**: `src/run_experiments.py`. This separately evaluates regression, classification, and association-rule experiments and writes their result tables to `data/`.

---

## Model Performance

The current live training pipeline compares Linear Regression, Random Forest, and Gradient Boosting, then selects the model with the highest held-out test $R^2$. The reproducible run in this repository selects **Random Forest**:

| Metric | Held-out test result |
| --- | ---: |
| $R^2$ | 0.9528 |
| MAE | 1.81 deg C |
| RMSE | 2.40 deg C |

The 5-fold cross-validation mean $R^2$ values are 0.8696 (Linear Regression), 0.8697 (Random Forest), and 0.8849 (Gradient Boosting). Cross-validation is reported for comparison, but it does not currently control model selection. These are dataset metrics, not validated hardware accuracy.

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

## Workflow

1. `src/data_preprocessing.py` removes duplicates and missing rows and creates ambient temperature, voltage, current load, and previous CPU temperature features.
2. `src/train_model.py` trains three regressors, reports test and cross-validation $R^2$, selects the best test score, and saves the model plus test set.
3. `src/evaluate_model.py` evaluates the saved artifact using MAE, RMSE, and $R^2$.
4. `app.py` provides manual predictions, risk labels, and optional Windows telemetry with prediction history.
5. `src/run_experiments.py` runs the separate chronological regression, classification, and Apriori experiments.

---

## 📈 Academic ML Laboratory Experiments

The original `src/train_model.py` and `models/server_temp_model.pkl` remain the live dashboard contract. The academic experiments use a separate artifact, `models/engineered_model_experiment.pkl`, and never silently replace the live model.

### Preprocessing

- Convert timestamps and numeric columns, sort chronologically, and remove duplicate rows.
- Keep missing CPU temperature targets out of supervised training; median-impute only input features.
- Detect constant columns and exclude them from experiments. The supplied CPU frequency column is constant.
- Inspect, rather than automatically delete, IQR outliers. The bounds are `Q1 - 1.5 * IQR` and `Q3 + 1.5 * IQR`.
- Demonstrate `MinMaxScaler` (`(x - min) / (max - min)`) and `StandardScaler` (`(x - mean) / standard deviation`). Scaling is used for KNN, SVR, and neural networks, but is unnecessary for tree models.
- Add previous CPU temperature, past rolling mean, CPU/temperature changes, and an approximately five-minute future target. Past features use `shift(1)` before rolling, so future values do not leak into current rows.

### Regression, classification, and association algorithms

Regression experiments include Simple Linear Regression, Multiple Linear Regression, Random Forest, AdaBoost, KNN, SVR, shallow neural network, deep neural network, and optional PCA comparison. Classification experiments use Random Forest and Gaussian Naive Bayes for `Safe`, `Moderate Risk`, and `High Risk`. A small Apriori implementation mines rules from discretized CPU usage, GPU temperature, and risk.

The current chronological regression run ranks Multiple Linear Regression first at $R^2=0.9426$, followed by Deep Neural Network at 0.9424 and Random Forest at 0.9406. The classification output reports 0.9884 accuracy and 0.9827 F1 for both Random Forest and Naive Bayes on the saved test window. The test window has limited minority-class representation, so these results should not be treated as production risk-detection validation. The association-rule output is descriptive, not a causal cooling policy.

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

`models/server_temp_model.pkl`, `models/engineered_model_experiment.pkl`, and the CSV outputs are generated artifacts. The live model is committed for convenience, but GitHub warns that it is approximately 94 MB; Git LFS is recommended for future model versions.

## Limitations and Next Steps

- Ambient temperature and voltage are synthetic/operator-provided inputs in the live model.
- Live prediction requires a previous CPU temperature sample and a GPU temperature reading from LibreHardwareMonitor.
- Dataset performance does not establish accuracy on unseen hardware.
- Improve temporal validation, collect representative high-risk samples, and validate against calibrated hardware sensors before operational use.

## Run the Project

Since the trained model file (`.pkl`) exceeds GitHub's size limit, please follow these steps to run the AI on your local machine:

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

### 4. Run the ML laboratory experiments

```powershell
python src/run_experiments.py
```

This runs chronological regression, classification, and Apriori experiments. It writes comparison tables to `data/regression_experiment_results.csv`, `data/classification_experiment_results.csv`, and `data/association_rules.csv`. It also saves the best separate engineered experiment artifact to `models/engineered_model_experiment.pkl`.

To inspect the dataset interactively with histograms, boxplots, correlation heatmap, CPU/temperature scatter plot, and IQR outlier counts:

```powershell
python src/eda.py
```

### 5. Run the dashboard
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

### 6. Command-line prediction (optional)
```powershell
python src/predict.py
```
This tests the saved model with a simulated high-load server scenario.

