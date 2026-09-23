"""Run a localhost-only dashboard instance for recording defense footage."""

from __future__ import annotations

import sys
from pathlib import Path

import uvicorn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from web.app import RuntimeState, create_app, load_camera_configs, load_settings


def main() -> None:
    settings_path = ROOT / "config/settings.yaml"
    cameras_dir = ROOT / "config/cameras"
    settings = load_settings(settings_path)
    settings.setdefault("web", {}).setdefault("auth", {})["enabled"] = False
    settings.setdefault("telegram", {})["enabled"] = False
    settings.setdefault("discord", {})["enabled"] = False
    settings.setdefault("siren", {})["enabled"] = False
    settings.setdefault("behavior_learning", {})["log_candidates"] = False
    cameras = load_camera_configs(cameras_dir)
    for camera in cameras.values():
        camera["notification_channels"] = []
    runtime = RuntimeState(settings, cameras, settings_path, cameras_dir)
    uvicorn.run(create_app(runtime), host="127.0.0.1", port=8765, log_level="warning")


if __name__ == "__main__":
    main()
