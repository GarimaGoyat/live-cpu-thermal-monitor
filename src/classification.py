"""Chronological CPU temperature risk classification experiments."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from preprocessing import BASE_FEATURES, ROOT, experiment_frame

RESULTS_PATH = ROOT / "data" / "classification_experiment_results.csv"
FEATURES = BASE_FEATURES + [
    "CPU_Temperature_Lag_1", "CPU_Temperature_Rolling_Mean_5",
    "CPU_Usage_Lag_1", "CPU_Temperature_Change", "CPU_Usage_Change",
]


def run_classification():
    frame = experiment_frame()
    features = [column for column in FEATURES if frame[column].nunique(dropna=False) > 1]
    first, second = int(len(frame) * 0.6), int(len(frame) * 0.8)
    train, validation, test = frame.iloc[:first], frame.iloc[first:second], frame.iloc[second:]
    X_train, X_test = train[features], test[features]
    X_validation = validation[features]
    y_train, y_test = train["Temperature_Risk"].astype(str), test["Temperature_Risk"].astype(str)
    y_validation = validation["Temperature_Risk"].astype(str)
    models = {
        "Random Forest Classifier": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(n_estimators=80, random_state=42, class_weight="balanced")),
        ]),
        "Naive Bayes Classifier": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", GaussianNB()),
        ]),
    }
    rows = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        validation_prediction = model.predict(X_validation)
        prediction = model.predict(X_test)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, prediction, average="weighted", zero_division=0)
        rows.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, prediction),
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "Validation_Accuracy": accuracy_score(y_validation, validation_prediction),
        })
        print(f"\n{name}")
        print(classification_report(y_test, prediction, zero_division=0))
        print("Confusion matrix (rows=actual, columns=predicted):")
        print(confusion_matrix(y_test, prediction, labels=["Safe", "Moderate Risk", "High Risk"]))
    results = pd.DataFrame(rows)
    RESULTS_PATH.parent.mkdir(exist_ok=True)
    results.to_csv(RESULTS_PATH, index=False)
    print("Classification results saved to", RESULTS_PATH)
    return results


if __name__ == "__main__":
    run_classification()
