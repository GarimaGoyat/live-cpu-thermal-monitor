import pandas as pd
import numpy as np


def clean_data(df):
    # Remove duplicates
    df = df.drop_duplicates()

    # Drop rows with missing values
    df = df.dropna()

    # --------- ENGINEERED FEATURES ---------

    # 1. Simulated Ambient Temperature (slow sinusoidal variation)
    df['Ambient_Temperature'] = 25 + 3 * np.sin(
        np.linspace(0, 2*np.pi, len(df))
    )

    # 2. Assume constant CPU core voltage
    df['Voltage'] = 12.0

    # 3. Approximate current load proportional to CPU usage
    df['Current_Load'] = df['CPU_Usage'] * 0.1

    # 4. Add previous CPU temperature (thermal inertia)
    df['Prev_CPU_Temperature'] = df['CPU_Temperature'].shift(1)

    # Drop first row (will contain NaN after shift)
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
