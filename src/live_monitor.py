"""Live Windows metrics and optional LibreHardwareMonitor sensor access."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import time
from typing import Optional

import psutil


@dataclass
class HardwareTemperatures:
    cpu_c: Optional[float] = None
    gpu_c: Optional[float] = None
    status: str = "LibreHardwareMonitor unavailable"


@dataclass
class LiveMetrics:
    timestamp: datetime
    cpu_usage: float
    memory_usage: float
    cpu_frequency: float
    disk_usage: float
    disk_read_mb_s: float
    disk_write_mb_s: float
    process_count: int
    thread_count: int
    network_sent_mb_s: float
    network_recv_mb_s: float
    cpu_temperature: Optional[float]
    gpu_temperature: Optional[float]
    sensor_status: str


def read_hardware_temperatures() -> HardwareTemperatures:
    """Read LibreHardwareMonitor's WMI sensors, without inventing values."""
    try:
        import wmi

        sensors = wmi.WMI(namespace="root\\LibreHardwareMonitor").Sensor()
    except Exception as exc:
        return HardwareTemperatures(status=f"Sensor unavailable: {exc}")

    cpu_values = []
    gpu_values = []
    for sensor in sensors:
        if str(getattr(sensor, "SensorType", "")) != "Temperature":
            continue
        try:
            value = float(sensor.Value)
        except (TypeError, ValueError):
            continue
        name = str(getattr(sensor, "Name", "")).lower()
        identifier = str(getattr(sensor, "Identifier", "")).lower()
        if "cpu" in name or "/cpu/" in identifier:
            cpu_values.append(value)
        elif "gpu" in name or "/gpu/" in identifier:
            gpu_values.append(value)

    if not cpu_values and not gpu_values:
        return HardwareTemperatures(
            status="No CPU/GPU temperature sensors found. Start LibreHardwareMonitor with WMI enabled."
        )
    return HardwareTemperatures(
        cpu_c=max(cpu_values) if cpu_values else None,
        gpu_c=max(gpu_values) if gpu_values else None,
        status="LibreHardwareMonitor connected",
    )


def _process_counts() -> tuple[int, int]:
    process_count = 0
    thread_count = 0
    for process in psutil.process_iter(["num_threads"]):
        try:
            process_count += 1
            thread_count += int(process.info.get("num_threads") or 0)
        except (psutil.Error, OSError, TypeError, ValueError):
            continue
    return process_count, thread_count


def collect_live_metrics(previous_io=None, interval_seconds: float = 1.0):
    """Collect one sample and return it with the I/O baseline for the next sample."""
    started = time.monotonic()
    cpu_usage = psutil.cpu_percent(interval=interval_seconds)
    memory_usage = psutil.virtual_memory().percent
    frequency = psutil.cpu_freq()
    disk = psutil.disk_usage(Path.cwd().anchor or "/")
    disk_io = psutil.disk_io_counters()
    network_io = psutil.net_io_counters()
    process_count, thread_count = _process_counts()

    disk_read_rate = 0.0
    disk_write_rate = 0.0
    network_sent_rate = 0.0
    network_recv_rate = 0.0
    if previous_io and disk_io and network_io:
        elapsed = max(started - previous_io["timestamp"], 0.001)
        disk_read_rate = max(0.0, disk_io.read_bytes - previous_io["disk_read"]) / elapsed / 1_048_576
        disk_write_rate = max(0.0, disk_io.write_bytes - previous_io["disk_write"]) / elapsed / 1_048_576
        network_sent_rate = max(0.0, network_io.bytes_sent - previous_io["network_sent"]) / elapsed / 1_048_576
        network_recv_rate = max(0.0, network_io.bytes_recv - previous_io["network_recv"]) / elapsed / 1_048_576

    hardware = read_hardware_temperatures()
    current_time = datetime.now(timezone.utc)
    next_io = {
        "timestamp": time.monotonic(),
        "disk_read": disk_io.read_bytes if disk_io else 0,
        "disk_write": disk_io.write_bytes if disk_io else 0,
        "network_sent": network_io.bytes_sent if network_io else 0,
        "network_recv": network_io.bytes_recv if network_io else 0,
    }
    return LiveMetrics(
        timestamp=current_time,
        cpu_usage=cpu_usage,
        memory_usage=memory_usage,
        cpu_frequency=frequency.current if frequency else 0.0,
        disk_usage=disk.percent,
        disk_read_mb_s=disk_read_rate,
        disk_write_mb_s=disk_write_rate,
        process_count=process_count,
        thread_count=thread_count,
        network_sent_mb_s=network_sent_rate,
        network_recv_mb_s=network_recv_rate,
        cpu_temperature=hardware.cpu_c,
        gpu_temperature=hardware.gpu_c,
        sensor_status=hardware.status,
    ), next_io