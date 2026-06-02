"""Entry point for the SCT real-time camera monitoring system."""

from __future__ import annotations

import os
import socket
from pathlib import Path

# Windows/OpenCV camera stability:
# - OBSENSOR can grab index 0 first and report "Camera index out of range".
# - MSMF hardware transforms can hang on some UVC webcams such as Logitech C270.
os.environ.setdefault("OPENCV_VIDEOIO_PRIORITY_OBSENSOR", "0")
os.environ.setdefault("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS", "0")
os.environ.setdefault(
    "OPENCV_FFMPEG_CAPTURE_OPTIONS",
    "rtsp_transport;tcp",
)

import uvicorn

from utils.logger import get_logger, setup_logging
from web.app import RuntimeState, create_app, load_camera_configs, load_settings

logger = get_logger(__name__)


def _local_ipv4_addresses() -> list[str]:
    """Return non-loopback IPv4 addresses for friendlier dashboard logs."""
    addresses: set[str] = set()
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ip_address = str(info[4][0])
            if not ip_address.startswith(("127.", "169.254.")):
                addresses.add(ip_address)
    except socket.gaierror:
        return []
    return sorted(addresses)


def _dashboard_urls(host: str, port: int) -> list[str]:
    if host in {"0.0.0.0", "::"}:
        urls = [f"http://localhost:{port}"]
        urls.extend(f"http://{ip_address}:{port}" for ip_address in _local_ipv4_addresses())
        return list(dict.fromkeys(urls))
    if host in {"127.0.0.1", "::1"}:
        return [f"http://localhost:{port}"]
    return [f"http://{host}:{port}"]


def main() -> None:
    """Load configuration, build the FastAPI app, and run Uvicorn."""
    project_root = Path(__file__).resolve().parent
    settings_path = project_root / "config" / "settings.yaml"
    cameras_dir = project_root / "config" / "cameras"

    settings = load_settings(settings_path)
    setup_logging(settings)
    cameras = load_camera_configs(cameras_dir)

    runtime = RuntimeState(
        settings=settings,
        cameras=cameras,
        settings_path=settings_path,
        cameras_dir=cameras_dir,
    )
    app = create_app(runtime)

    web_settings = settings.get("web", {})
    host = str(web_settings.get("host", "0.0.0.0"))
    port = int(web_settings.get("port", 8000))
    for url in _dashboard_urls(host, port):
        logger.info("Dashboard URL: %s", url)
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_config=None,
    )


if __name__ == "__main__":
    main()
