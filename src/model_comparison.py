"""Chronological regression experiments for the ML laboratory."""

from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

sys.path.insert(0, str(Path(__file__).parent))
from preprocessing import BASE_FEATURES, ROOT, experiment_frame, fit_input_preprocessor

MODEL_PATH = ROOT / "models" / "engineered_model_experiment.pkl"
RESULTS_PATH = ROOT / "data" / "regression_experiment_results.csv"
ENGINEERED_FEATURES = [
    "CPU_Temperature_Lag_1", "CPU_Temperature_Rolling_Mean_5",
    "CPU_Usage_Lag_1", "CPU_Temperature_Change", "CPU_Usage_Change",
]


def chronological_split(frame):
    first = int(len(frame) * 0.6)
    second = int(len(frame) * 0.8)
    return frame.iloc[:first], frame.iloc[first:second], frame.iloc[second:]


def metrics(y_true, prediction):
    return {
        "MAE": mean_absolute_error(y_true, prediction),
        "RMSE": np.sqrt(mean_squared_error(y_true, prediction)),
        "R2": r2_score(y_true, prediction),
    }


def imputed_model(model, scale=False):
    steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", model))
    return Pipeline(steps)


def evenly_sample(frame, maximum=8000):
    if len(frame) <= maximum:
        return frame
    indexes = np.linspace(0, len(frame) - 1, maximum, dtype=int)
    return frame.iloc[indexes]


def run_experiments():
    frame = experiment_frame()
    feature_columns = BASE_FEATURES + ENGINEERED_FEATURES
    feature_columns = [column for column in feature_columns if frame[column].nunique(dropna=False) > 1]
    train, validation, test = chronological_split(frame)
    X_train, y_train = train[feature_columns], train["CPU_Temperature"]
    X_validation, y_validation = validation[feature_columns], validation["CPU_Temperature"]
    X_test, y_test = test[feature_columns], test["CPU_Temperature"]
    minmax_imputer, minmax_scaler = fit_input_preprocessor(X_train, scale="minmax")
    standard_imputer, standard_scaler = fit_input_preprocessor(X_train, scale="standard")
    minmax_values = minmax_scaler.transform(minmax_imputer.transform(X_train))
    standard_values = standard_scaler.transform(standard_imputer.transform(X_train))
    print(f"MinMaxScaler range on training data: {minmax_values.min():.1f} to {minmax_values.max():.1f}")
    print(f"StandardScaler mean (first feature): {standard_values[:, 0].mean():.3f}")

    models = {
        "Simple Linear Regression": (imputed_model(LinearRegression()), ["CPU_Usage"]),
        "Multiple Linear Regression": (imputed_model(LinearRegression()), feature_columns),
        "Random Forest Regressor": (imputed_model(RandomForestRegressor(n_estimators=80, random_state=42, n_jobs=-1)), feature_columns),
        "AdaBoost Regressor": (imputed_model(AdaBoostRegressor(n_estimators=50, random_state=42)), feature_columns),
        "KNN Regressor": (imputed_model(KNeighborsRegressor(n_neighbors=5), scale=True), feature_columns),
        "Support Vector Regression": (imputed_model(SVR(kernel="rbf", C=10), scale=True), feature_columns),
        "Shallow Neural Network": (imputed_model(MLPRegressor(hidden_layer_sizes=(32,), max_iter=150, random_state=42)), feature_columns),
        "Deep Neural Network": (imputed_model(MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=150, random_state=42)), feature_columns),
    }

    slow_models = {"KNN Regressor", "Support Vector Regression", "Shallow Neural Network", "Deep Neural Network"}
    rows = []
    fitted = {}
    for name, (model, columns) in models.items():
        fit_frame = evenly_sample(train, 8000) if name in slow_models else train
        model.fit(fit_frame[columns], fit_frame["CPU_Temperature"])
        validation_prediction = model.predict(X_validation[columns])
        test_prediction = model.predict(X_test[columns])
        result = metrics(y_test, test_prediction)
        result.update({"Model": name, "Validation_R2": r2_score(y_validation, validation_prediction), "Feature_Count": len(columns)})
        rows.append(result)
        fitted[name] = (model, columns)
        print(f"{name}: MAE={result['MAE']:.3f}, RMSE={result['RMSE']:.3f}, R2={result['R2']:.4f}")

    importance_model = fitted["Random Forest Regressor"][0]
    importance_values = importance_model.named_steps["model"].feature_importances_
    importance = pd.DataFrame({"Feature": feature_columns, "Importance": importance_values}).sort_values("Importance", ascending=False)
    print("\nRandom Forest feature importance:")
    print(importance.to_string(index=False))

    selected = importance.head(max(3, len(importance) // 2))["Feature"].tolist()
    selected_model = imputed_model(RandomForestRegressor(n_estimators=80, random_state=42, n_jobs=-1))
    selected_model.fit(train[selected], y_train)
    selected_prediction = selected_model.predict(X_test[selected])
    selected_result = metrics(y_test, selected_prediction)
    selected_result.update({"Model": "Random Forest (selected features)", "Validation_R2": r2_score(y_validation, selected_model.predict(X_validation[selected])), "Feature_Count": len(selected)})
    rows.append(selected_result)

    pca_model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=0.95)),
        ("model", LinearRegression()),
    ])
    pca_model.fit(train[feature_columns], y_train)
    pca_prediction = pca_model.predict(X_test[feature_columns])
    pca_result = metrics(y_test, pca_prediction)
    pca_result.update({"Model": "Linear Regression with PCA", "Validation_R2": r2_score(y_validation, pca_model.predict(X_validation[feature_columns])), "Feature_Count": int(pca_model.named_steps["pca"].n_components_)})
    rows.append(pca_result)
    print(f"PCA components explaining 95% variance: {pca_result['Feature_Count']}")

    results = pd.DataFrame(rows).sort_values("R2", ascending=False)
    RESULTS_PATH.parent.mkdir(exist_ok=True)
    results.to_csv(RESULTS_PATH, index=False)
    best_name = results.iloc[0]["Model"]
    if best_name == "Random Forest (selected features)":
        best_model, best_columns = selected_model, selected
    elif best_name == "Linear Regression with PCA":
        best_model, best_columns = pca_model, feature_columns
    else:
        best_model, best_columns = fitted[best_name]
    joblib.dump({"model": best_model, "features": best_columns, "target": "CPU_Temperature"}, MODEL_PATH)
    print("\nResults saved to", RESULTS_PATH)
    print("Separate experiment model saved to", MODEL_PATH)
    print(results[["Model", "MAE", "RMSE", "R2", "Validation_R2"]].to_string(index=False))


if __name__ == "__main__":
    run_experiments()
