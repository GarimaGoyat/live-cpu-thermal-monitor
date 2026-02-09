import pandas as pd
import numpy as np


def clean_data(df):
    """
    Clean and preprocess server temperature data.
    
    Steps:
    1. Remove duplicates
    2. Drop rows with missing values
    3. Engineer synthetic features for environmental and thermal factors
    
    Returns:
        DataFrame with 11 features ready for model training
    """
    # Remove duplicates
    df = df.drop_duplicates()

    # Drop rows with missing values (ensures data consistency)
    df = df.dropna()

    # --------- ENGINEERED FEATURES ---------
    # These synthetic features capture environmental and thermal dynamics
    
    # 1. Ambient Temperature - Simulated with sinusoidal variation (25±3°C)
    #    Captures realistic daily thermal cycles
    df['Ambient_Temperature'] = 25 + 3 * np.sin(
        np.linspace(0, 2*np.pi, len(df))
    )

    # 2. Voltage - Constant CPU core voltage (12.0V)
    #    Represents stable power supply assumption
    df['Voltage'] = 12.0

    # 3. Current Load - Approximate electrical load from CPU usage
    #    Formula: CPU_Usage × 0.1
    df['Current_Load'] = df['CPU_Usage'] * 0.1

    # 4. Previous CPU Temperature - Lagged temperature value
    #    Captures thermal inertia and temporal dependencies (t-1)
    df['Prev_CPU_Temperature'] = df['CPU_Temperature'].shift(1)

    # Drop first row (will contain NaN after shift) and any other NaN values
    df = df.dropna()

    return df


# Optional standalone cleaning runner
if __name__ == "__main__":
    try:
        raw_data = pd.read_csv('data/server_temperature_data.csv')
        cleaned_df = clean_data(raw_data)
        print(f"Cleaning complete. Remaining rows: {len(cleaned_df)}")
        cleaned_df.to_csv('data/cleaned_server_data.csv', index=False)
        print("File saved as 'data/cleaned_server_data.csv'")
    except FileNotFoundError:
        print("Error: Could not find the CSV file. Check your folder path!")
