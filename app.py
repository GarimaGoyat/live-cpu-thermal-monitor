from pathlib import Path
import sys

import joblib
import pandas as pd
import streamlit as st
from streamlit_autorefresh import st_autorefresh

sys.path.insert(0, str(Path(__file__).parent / "src"))
from live_monitor import collect_live_metrics

FEATURES = [
    "CPU_Usage", "CPU_Frequency", "Memory_Usage", "Disk_Usage",
    "Process_Count", "Thread_Count", "GPU_Temperature",
    "Ambient_Temperature", "Voltage", "Current_Load", "Prev_CPU_Temperature",
]
ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "models" / "server_temp_model.pkl"
LOG_PATH = ROOT / "data" / "live_monitoring_log.csv"


@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    model_features = getattr(model, "feature_names_in_", None)
    if model_features is not None and list(model_features) != FEATURES:
        raise ValueError(f"Model feature order does not match training contract: {list(model_features)}")
    return model


def risk_label(temperature: float) -> str:
    return "Safe" if temperature < 70 else "Moderate Risk" if temperature < 80 else "High Risk"


def append_log(row: dict):
    pd.DataFrame([row]).to_csv(LOG_PATH, mode="a", header=not LOG_PATH.exists(), index=False)


def show_prediction(prediction: float, actual: float | None = None):
    st.metric("Predicted CPU temperature", f"{prediction:.2f} °C", risk_label(prediction))
    if actual is None:
        st.info("Actual CPU temperature is unavailable until LibreHardwareMonitor reports a sensor value.")
    else:
        st.metric("Actual CPU temperature", f"{actual:.2f} °C", f"error {prediction - actual:+.2f} °C")


def live_input(metrics, ambient_temperature: float, voltage: float, previous_temperature: float):
    return pd.DataFrame([{
        "CPU_Usage": metrics.cpu_usage,
        "CPU_Frequency": metrics.cpu_frequency,
        "Memory_Usage": metrics.memory_usage,
        "Disk_Usage": metrics.disk_usage,
        "Process_Count": metrics.process_count,
        "Thread_Count": metrics.thread_count,
        "GPU_Temperature": metrics.gpu_temperature,
        "Ambient_Temperature": ambient_temperature,
        "Voltage": voltage,
        "Current_Load": metrics.cpu_usage * 0.1,
        "Prev_CPU_Temperature": previous_temperature,
    }], columns=FEATURES)


st.set_page_config(page_title="CPU Thermal Monitor", layout="wide")
st.title("CPU Thermal Monitor")
st.caption("Manual prediction and measured Windows live monitoring")

try:
    model = load_model()
except FileNotFoundError:
    st.error("Model not found. Run `python src/train_model.py` first.")
    st.stop()

with st.sidebar:
    st.header("Model and risk")
    st.caption("Run `python src/evaluate_model.py` for current held-out metrics. Results are dataset metrics, not validated hardware accuracy.")
    st.markdown("- Safe: < 70 °C\n- Moderate Risk: 70 to < 80 °C\n- High Risk: >= 80 °C")
    mode = st.radio("Mode", ["Manual Prediction", "Live Monitoring"])

if mode == "Manual Prediction":
    st.subheader("Manual prediction")
    left, right = st.columns(2)
    with left:
        cpu_usage = st.slider("CPU Usage (%)", 0.0, 100.0, 50.0)
        cpu_frequency = st.number_input("CPU Frequency (MHz)", 0.0, 10000.0, 3500.0)
        memory_usage = st.slider("Memory Usage (%)", 0.0, 100.0, 40.0)
        process_count = st.number_input("Process Count", 0, 10000, 200)
        thread_count = st.number_input("Thread Count", 0, 100000, 2500)
    with right:
        disk_usage = st.slider("Disk Usage (%)", 0.0, 100.0, 50.0)
        gpu_temperature = st.number_input("GPU Temperature (°C)", 0.0, 150.0, 40.0)
        ambient_temperature = st.number_input("Ambient Temperature (°C)", -50.0, 100.0, 25.0)
        voltage = st.number_input("Voltage (V)", 0.0, 30.0, 12.0)
        previous_temperature = st.number_input("Previous CPU Temperature (°C)", 0.0, 150.0, 50.0)
    if st.button("Predict temperature", type="primary"):
        manual = pd.DataFrame([{
            "CPU_Usage": cpu_usage, "CPU_Frequency": cpu_frequency,
            "Memory_Usage": memory_usage, "Disk_Usage": disk_usage,
            "Process_Count": process_count, "Thread_Count": thread_count,
            "GPU_Temperature": gpu_temperature, "Ambient_Temperature": ambient_temperature,
            "Voltage": voltage, "Current_Load": cpu_usage * 0.1,
            "Prev_CPU_Temperature": previous_temperature,
        }], columns=FEATURES)
        show_prediction(float(model.predict(manual)[0]))
else:
    interval = st.sidebar.slider("Refresh interval (seconds)", 2, 5, 3)
    ambient_temperature = st.sidebar.number_input("Ambient temperature (°C)", -50.0, 100.0, 25.0)
    voltage = st.sidebar.number_input("Voltage (V)", 0.0, 30.0, 12.0)
    log_samples = st.sidebar.checkbox("Log samples to data/live_monitoring_log.csv")
    st_autorefresh(interval=interval * 1000, key="live-refresh")
    st.session_state.setdefault("live_io", None)
    st.session_state.setdefault("live_history", [])
    metrics, st.session_state.live_io = collect_live_metrics(st.session_state.live_io, interval_seconds=1.0)
    previous_temperature = st.session_state.get("live_previous_cpu_temperature")

    st.subheader("Live monitoring")
    st.caption(f"Sample: {metrics.timestamp.astimezone().strftime('%Y-%m-%d %H:%M:%S %Z')} | {metrics.sensor_status}")
    columns = st.columns(6)
    values = [
        ("CPU", f"{metrics.cpu_usage:.1f}%"), ("Memory", f"{metrics.memory_usage:.1f}%"),
        ("CPU frequency", f"{metrics.cpu_frequency:.0f} MHz"), ("Disk capacity", f"{metrics.disk_usage:.1f}%"),
        ("Processes", str(metrics.process_count)), ("Threads", str(metrics.thread_count)),
    ]
    for column, (label, value) in zip(columns, values):
        column.metric(label, value)
    columns = st.columns(4)
    values = [
        ("Disk read", f"{metrics.disk_read_mb_s:.2f} MB/s"), ("Disk write", f"{metrics.disk_write_mb_s:.2f} MB/s"),
        ("Network sent", f"{metrics.network_sent_mb_s:.2f} MB/s"), ("Network received", f"{metrics.network_recv_mb_s:.2f} MB/s"),
    ]
    for column, (label, value) in zip(columns, values):
        column.metric(label, value)

    if metrics.gpu_temperature is None or previous_temperature is None:
        st.warning("Live prediction is unavailable: actual GPU temperature and a previous CPU temperature reading are required from LibreHardwareMonitor.")
    else:
        inputs = live_input(metrics, ambient_temperature, voltage, previous_temperature)
        prediction = float(model.predict(inputs)[0])
        result_columns = st.columns(2)
        with result_columns[0]:
            show_prediction(prediction, metrics.cpu_temperature)
        with result_columns[1]:
            st.metric("GPU temperature", f"{metrics.gpu_temperature:.2f} °C")
        row = {
            "Timestamp": metrics.timestamp.isoformat(), **inputs.iloc[0].to_dict(),
            "Predicted_CPU_Temperature": prediction, "Actual_CPU_Temperature": metrics.cpu_temperature,
            "Disk_Read_MB_s": metrics.disk_read_mb_s, "Disk_Write_MB_s": metrics.disk_write_mb_s,
            "Network_Sent_MB_s": metrics.network_sent_mb_s, "Network_Recv_MB_s": metrics.network_recv_mb_s,
            "Risk": risk_label(prediction),
        }
        st.session_state.live_history = (st.session_state.live_history + [row])[-120:]
        if log_samples:
            append_log(row)

    if metrics.cpu_temperature is not None:
        st.session_state.live_previous_cpu_temperature = metrics.cpu_temperature
    if st.session_state.live_history:
        history = pd.DataFrame(st.session_state.live_history).set_index("Timestamp")
        st.subheader("Temperature over time")
        chart_columns = ["Predicted_CPU_Temperature"]
        if history["Actual_CPU_Temperature"].notna().any():
            chart_columns.append("Actual_CPU_Temperature")
        st.line_chart(history[chart_columns])
    with st.expander("Feature contract"):
        st.write("The model receives the original 11 columns in the original order.")
        st.code(", ".join(FEATURES))
        st.write("Ambient temperature and voltage are operator-provided because the original training pipeline synthesized them. Current load remains CPU usage × 0.1 to match training. Disk I/O and network rates are display-only.")
