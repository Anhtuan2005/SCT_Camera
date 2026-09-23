from scripts.monitor_runtime import _camera_values, _parse_gpu_line


def test_parse_gpu_line() -> None:
    metrics = _parse_gpu_line("NVIDIA RTX 3050 Laptop GPU, 42, 1024, 4096, 71, 48.5, 1500")

    assert metrics == {
        "gpu_name": "NVIDIA RTX 3050 Laptop GPU",
        "gpu_util_percent": 42.0,
        "vram_used_mb": 1024.0,
        "vram_total_mb": 4096.0,
        "gpu_temperature_c": 71.0,
        "gpu_power_w": 48.5,
        "gpu_clock_mhz": 1500.0,
    }


def test_camera_values() -> None:
    values = _camera_values(
        [
            {
                "camera_id": "cam_01",
                "status": "online",
                "fps": 24.5,
                "capture_fps": 25.0,
                "ai_fps": 9.5,
                "dropped_capture_frames": 3,
            }
        ],
        ["cam_01"],
    )

    assert values["online_cameras"] == 1
    assert values["aggregate_fps"] == 24.5
    assert values["aggregate_ai_fps"] == 9.5
    assert values["cam_01_dropped_frames"] == 3
