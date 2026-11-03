"""Exploratory analysis plots for the server temperature dataset."""

import matplotlib.pyplot as plt
import seaborn as sns

from preprocessing import experiment_frame, load_raw_data


def print_summary(frame):
    print(f"Dataset shape: {frame.shape}")
    print("\nData types:\n", frame.dtypes)
    print("\nMissing values:\n", frame.isna().sum())
    print("\nStatistical summary:\n", frame.describe(numeric_only=True).round(2))


def iqr_outlier_summary(frame, columns):
    rows = []
    for column in columns:
        values = frame[column].dropna()
        first_quartile = values.quantile(0.25)
        third_quartile = values.quantile(0.75)
        iqr = third_quartile - first_quartile
        lower = first_quartile - 1.5 * iqr
        upper = third_quartile + 1.5 * iqr
        rows.append({
            "Feature": column,
            "Lower_Bound": lower,
            "Upper_Bound": upper,
            "Potential_Outliers": int(((values < lower) | (values > upper)).sum()),
        })
    return rows


def show_eda(frame):
    numeric_columns = frame.select_dtypes(include="number").columns.tolist()
    print_summary(frame)
    print("\nIQR potential outliers (not automatically deleted):")
    for row in iqr_outlier_summary(frame, numeric_columns):
        print(row)

    frame[numeric_columns].hist(figsize=(14, 10), bins=30)
    plt.suptitle("Numerical Feature Histograms")
    plt.tight_layout()

    plt.figure(figsize=(14, 7))
    sns.boxplot(data=frame[numeric_columns], orient="h")
    plt.title("Boxplots for IQR Outlier Inspection")
    plt.tight_layout()

    plt.figure(figsize=(10, 7))
    sns.heatmap(frame[numeric_columns].corr(), annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()

    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=frame, x="CPU_Usage", y="CPU_Temperature", alpha=0.25)
    plt.title("CPU Usage vs CPU Temperature")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    show_eda(experiment_frame())
