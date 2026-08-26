"""Record long-running SCT Camera health metrics to CSV."""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
import psutil
import yaml


ROOT = Path(__file__).resolve().parents[1]
GPU_FIELDS = (
    "gpu_name",
    "gpu_util_percent",
    "vram_used_mb",
    "vram_total_mb",
    "gpu_temperature_c",
    "gpu_power_w",
    "gpu_clock_mhz",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pid", type=int, help="PID of the Python process running main.py")
    parser.add_argument("--interval", type=float, default=10.0, help="Seconds between samples")
    parser.add_argument("--duration", type=float, default=14_400.0, help="Total seconds to record")
    parser.add_argument("--out", type=Path, required=True, help="Destination CSV file")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--settings", type=Path, default=ROOT / "config/settings.yaml")
    args = parser.parse_args()
    if args.interval <= 0 or args.duration <= 0:
        parser.error("--interval and --duration must be positive")
    return args


def _find_app_process(pid: int | None) -> psutil.Process:
    if pid is not None:
        try:
            return psutil.Process(pid)
        except psutil.Error as exc:
            raise SystemExit(f"Cannot monitor PID {pid}: {exc}") from exc

    candidates: list[psutil.Process] = []
    for process in psutil.process_iter(("pid", "cmdline")):
        try:
            command = process.info.get("cmdline") or []
            if any(str(part).replace("\\", "/").rsplit("/", 1)[-1].lower() == "main.py" for part in command):
                candidates.append(process)
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            continue

    if not candidates:
        raise SystemExit("Cannot find main.py. Start `python main.py` first, or pass --pid.")
    if len(candidates) > 1:
        candidate_ids = {process.pid for process in candidates}
        leaves = [process for process in candidates if not any(other.ppid() == process.pid for other in candidates)]
        if len(leaves) == 1 and leaves[0].ppid() in candidate_ids:
            return leaves[0]
        pids = ", ".join(str(process.pid) for process in candidates)
        raise SystemExit(f"Multiple independent main.py processes found ({pids}); pass --pid.")
    return candidates[0]


def _number(value: str) -> float | None:
    try:
        return float(value.strip())
    except ValueError:
        return None


def _parse_gpu_line(line: str) -> dict[str, Any]:
    values = next(csv.reader([line]))
    if len(values) != len(GPU_FIELDS):
        return {field: None for field in GPU_FIELDS}
    parsed: dict[str, Any] = {"gpu_name": values[0].strip()}
    parsed.update({field: _number(value) for field, value in zip(GPU_FIELDS[1:], values[1:])})
    return parsed


def _gpu_snapshot() -> dict[str, Any]:
    query = ",".join(
        (
            "name",
            "utilization.gpu",
            "memory.used",
            "memory.total",
            "temperature.gpu",
            "power.draw",
            "clocks.current.graphics",
        )
    )
    try:
        result = subprocess.run(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        return _parse_gpu_line(result.stdout.splitlines()[0])
    except (FileNotFoundError, IndexError, subprocess.SubprocessError):
        return {field: None for field in GPU_FIELDS}


def _auth_settings(path: Path) -> tuple[bool, str, str | None]:
    settings = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    auth = settings.get("web", {}).get("auth", {})
    if not isinstance(auth, dict):
        return bool(auth), "admin", None
    username = os.getenv(str(auth.get("username_env") or "SCT_CAMERA_USERNAME")) or auth.get("username") or "admin"
    password = os.getenv(str(auth.get("password_env") or "SCT_CAMERA_PASSWORD")) or auth.get("password")
    return bool(auth.get("enabled", False)), str(username), str(password) if password is not None else None


class CameraClient:
    def __init__(self, base_url: str, settings_path: Path) -> None:
        self.client = httpx.Client(base_url=base_url.rstrip("/"), timeout=5)
        self.auth_enabled, self.username, self.password = _auth_settings(settings_path)
        if self.auth_enabled:
            self._login()

    def _login(self) -> None:
        if self.password is None:
            raise SystemExit("Dashboard authentication is enabled but no plaintext password is available.")
        response = self.client.post(
            "/login",
            data={"username": self.username, "password": self.password, "next": "/"},
        )
        if response.status_code not in {302, 303}:
            raise SystemExit(f"Dashboard login failed: HTTP {response.status_code}")

    def cameras(self) -> list[dict[str, Any]]:
        response = self.client.get("/api/cameras")
        if response.status_code == 401 and self.auth_enabled:
            self._login()
            response = self.client.get("/api/cameras")
        response.raise_for_status()
        cameras = response.json()
        if not isinstance(cameras, list):
            raise RuntimeError("/api/cameras did not return a list")
        return cameras

    def close(self) -> None:
        self.client.close()


def _camera_fields(camera_ids: list[str]) -> list[str]:
    return [
        f"{camera_id}_{metric}"
        for camera_id in camera_ids
        for metric in ("fps", "capture_fps", "ai_fps", "dropped_frames")
    ]


def _camera_values(cameras: list[dict[str, Any]], camera_ids: list[str]) -> dict[str, Any]:
    by_id = {str(camera.get("camera_id")): camera for camera in cameras}
    online = [camera for camera in cameras if camera.get("status") == "online"]

    values: dict[str, Any] = {
        "camera_count": len(cameras),
        "online_cameras": len(online),
        "aggregate_fps": round(
            sum(float(camera.get("fps") or 0) for camera in online), 3
        ),
        "aggregate_ai_fps": round(
            sum(float(camera.get("ai_fps") or 0) for camera in online), 3
        ),
    }

    for camera_id in camera_ids:
        camera = by_id.get(camera_id, {})
        values[f"{camera_id}_fps"] = camera.get("fps")
        values[f"{camera_id}_capture_fps"] = camera.get("capture_fps")
        values[f"{camera_id}_ai_fps"] = camera.get("ai_fps")
        values[f"{camera_id}_dropped_frames"] = camera.get("dropped_capture_frames")

    return values


def main() -> int:
    args = _parse_args()
    process = _find_app_process(args.pid)
    camera_client = CameraClient(args.base_url, args.settings)
    try:
        initial_cameras = camera_client.cameras()
    except (httpx.HTTPError, RuntimeError) as exc:
        camera_client.close()
        raise SystemExit(f"Cannot read {args.base_url}/api/cameras: {exc}") from exc

    camera_ids = sorted(str(camera["camera_id"]) for camera in initial_cameras if camera.get("camera_id"))
    fields = [
        "timestamp",
        "elapsed_seconds",
        "process_pid",
        "process_cpu_percent",
        "process_ram_mb",
        "process_threads",
        "system_ram_percent",
        *GPU_FIELDS,
        "camera_count",
        "online_cameras",
        "aggregate_fps",
        "aggregate_ai_fps",
        *_camera_fields(camera_ids),
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    process.cpu_percent(None)
    started = time.monotonic()
    next_sample = started
    rows = 0

    print(f"Monitoring PID {process.pid}; writing {args.out}")
    try:
        with args.out.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            while time.monotonic() - started < args.duration:
                if not process.is_running():
                    raise RuntimeError(f"Process {process.pid} stopped")
                cameras = camera_client.cameras()
                memory = process.memory_info()
                elapsed = time.monotonic() - started
                row = {
                    "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
                    "elapsed_seconds": round(elapsed, 3),
                    "process_pid": process.pid,
                    "process_cpu_percent": round(process.cpu_percent(None), 2),
                    "process_ram_mb": round(memory.rss / (1024 * 1024), 3),
                    "process_threads": process.num_threads(),
                    "system_ram_percent": psutil.virtual_memory().percent,
                    **_gpu_snapshot(),
                    **_camera_values(cameras, camera_ids),
                }
                writer.writerow(row)
                handle.flush()
                rows += 1
                print(
                    f"\r{elapsed:8.0f}s | RAM {row['process_ram_mb']:8.1f} MB | "
                    f"VRAM {row['vram_used_mb'] or 0:7.0f} MB | AI FPS {row['aggregate_ai_fps']:6.1f}",
                    end="",
                    flush=True,
                )
                next_sample += args.interval
                time.sleep(max(0.0, min(next_sample - time.monotonic(), args.duration - elapsed)))
    except KeyboardInterrupt:
        pass
    finally:
        camera_client.close()
    print(f"\nRecorded {rows} samples to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
