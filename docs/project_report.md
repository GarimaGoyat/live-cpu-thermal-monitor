
# 📝 Project Report: Server CPU Temperature Prediction

### 1. Problem Statement

High-density server environments often face overheating issues that lead to hardware throttling or permanent failure. This project implements a **Random Forest Regressor** to predict CPU temperatures based on real-time resource utilization metrics.

### 2. Data Preprocessing

The raw dataset required extensive cleaning to ensure model reliability:

* **Missing Value Management**: Successfully handled **327 missing sensor records** using mean imputation.
* **Feature Scaling**: Normalized metrics including CPU usage and memory to optimize the Random Forest algorithm.

![Data Preprocessing Success Log](data_preprocessing.png)

### 3. Technical Implementation

* **Language**: Python 3.12.
* **Architecture**: Modular project structure with separate directories for data, models, and source code.
* **Environment**: Isolated virtual environment (`.venv`) to manage dependencies like Scikit-Learn and Pandas.

### 4. Model Training & Evaluation Results

- **Train/Test Split**: The dataset is split with `train_test_split(..., test_size=0.2, random_state=42)`. The training script saves the held-out test set to `data/test_set.csv` for reproducible evaluation.
- **Saved Artifacts**:
  - Model: `models/server_temp_model.pkl` (saved by `src/train_model.py`)
  - Test set: `data/test_set.csv`
- **Evaluation Metrics (on held-out test set)**:
  - Mean Absolute Error (MAE): **2.33°C**
  - R² Score: **0.9174**

*Evaluation screenshot:* `docs/model_accuracy_report.png`

![Evaluation Results Screenshot](model_accuracy_report.png)

### 5. Conclusion

The project successfully bridges the gap between raw hardware metrics and actionable thermal insights. By training with a held-out test split and saving the test set for consistent evaluation, the model achieved strong generalization on the held-out set (MAE 2.33°C, R² 0.9174). Next steps include cross-validation, hyperparameter tuning, and adding CI checks to prevent performance regressions.
