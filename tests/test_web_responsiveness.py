from __future__ import annotations

import asyncio
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

import httpx
from fastapi.testclient import TestClient

from core.frame_buffer import FrameBuffer
from web.app import create_app
from web.routes.stream import camera_stream


class _AlertManager:
    _running = True


class _Runtime:
    def __init__(self, camera_count: int = 0, upsert_delay: float = 0.0) -> None:
        self.settings = {"web": {"auth": {"enabled": False}}}
        self.alert_manager = _AlertManager()
        self.upsert_delay = upsert_delay
        self.buffers = {f"cam{index}": FrameBuffer() for index in range(camera_count)}

    async def start(self) -> None:
        return None

    async def stop(self) -> None:
        return None

    def list_cameras(self) -> list[dict[str, object]]:
        return [self.get_camera(camera_id) for camera_id in self.buffers]

    def get_camera(self, camera_id: str) -> dict[str, object] | None:
        if camera_id not in self.buffers:
            return None
        return {
            "camera_id": camera_id,
            "name": camera_id,
            "enabled": True,
            "status": "online",
            "object_count": 0,
            "alert_count": 0,
        }

    def frame_buffer(self, camera_id: str) -> FrameBuffer | None:
        return self.buffers.get(camera_id)

    def upsert_camera(self, payload: dict[str, object]) -> dict[str, object]:
        time.sleep(self.upsert_delay)
        return payload


class WebResponsivenessTests(unittest.TestCase):
    def test_recent_alerts_distinguish_missing_assets_from_possible_theft(self) -> None:
        root = Path(__file__).parents[1]
        javascript = (root / "web" / "static" / "js" / "main.js").read_text(
            encoding="utf-8"
        )
        base_template = (root / "web" / "templates" / "base.html").read_text(
            encoding="utf-8"
        )

        self.assertIn('asset_missing: "Asset missing"', javascript)
        self.assertIn('asset_removed: "Asset removed"', javascript)
        self.assertIn('suspicious_theft_behavior: "Possible theft"', javascript)
        self.assertIn("main.js?v=security-console-15", base_template)

    def test_camera_detail_keeps_two_columns_until_compact_breakpoint(self) -> None:
        css = (Path(__file__).parents[1] / "web" / "static" / "css" / "style.css").read_text(
            encoding="utf-8"
        )
        javascript = (Path(__file__).parents[1] / "web" / "static" / "js" / "main.js").read_text(
            encoding="utf-8"
        )
        desktop_detail_css = css.split(".detail-layout {", 1)[1].split("}", 1)[0]
        portrait_detail_css = css.split(".detail-layout.is-portrait {", 1)[1].split("}", 1)[0]
        responsive_detail_css = css.split("@media (max-width: 1180px)", 1)[1].split(
            "@media (max-width: 900px)", 1
        )[0]

        self.assertIn("clamp(300px, 24vw, 380px)", desktop_detail_css)
        self.assertIn("minmax(460px, 1.2fr)", portrait_detail_css)
        self.assertIn(".detail-layout", responsive_detail_css)
        self.assertIn("grid-template-columns: minmax(0, 1fr);", responsive_detail_css)
        self.assertIn(".detail-side", responsive_detail_css)
        self.assertIn('detail.classList.toggle("is-portrait"', javascript)
        self.assertIn("stream.naturalHeight > stream.naturalWidth", javascript)

    def test_portrait_camera_uses_bounded_zoom_and_aligned_metrics(self) -> None:
        root = Path(__file__).parents[1]
        css = (root / "web" / "static" / "css" / "style.css").read_text(encoding="utf-8")
        javascript = (root / "web" / "static" / "js" / "main.js").read_text(encoding="utf-8")
        template = (root / "web" / "templates" / "camera_detail.html").read_text(encoding="utf-8")

        self.assertIn('class="detail-metrics"', template)
        self.assertIn("grid-template-columns: repeat(4, minmax(0, 1fr));", css)
        self.assertIn("fitScale * 1.35", javascript)
        self.assertIn('height > width ? "Zoom" : "Fill"', javascript)

    def test_mjpeg_stream_disables_intermediary_buffering(self) -> None:
        runtime = _Runtime(camera_count=1)
        request = SimpleNamespace(
            app=SimpleNamespace(state=SimpleNamespace(runtime=runtime))
        )

        response = camera_stream(request, "cam0")

        self.assertEqual("no-store, no-cache, must-revalidate, max-age=0", response.headers["cache-control"])
        self.assertEqual("no-cache", response.headers["pragma"])
        self.assertEqual("no", response.headers["x-accel-buffering"])

    def test_dashboard_uses_mjpeg_streams_for_realtime_video(self) -> None:
        client = TestClient(create_app(_Runtime(camera_count=6)))

        response = client.get("/")

        self.assertEqual(200, response.status_code)
        self.assertEqual(6, response.text.count('/api/stream/'))

    def test_snapshot_returns_a_jpeg_without_holding_the_connection(self) -> None:
        client = TestClient(create_app(_Runtime(camera_count=1)))

        response = client.get("/api/snapshot/cam0")

        self.assertEqual(200, response.status_code)
        self.assertEqual("image/jpeg", response.headers["content-type"])
        self.assertEqual("no-store", response.headers["cache-control"])
        self.assertTrue(response.content.startswith(b"\xff\xd8"))

    def test_slow_camera_mutation_does_not_block_other_requests(self) -> None:
        async def exercise() -> float:
            app = create_app(_Runtime(upsert_delay=0.25))
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                started = time.perf_counter()
                slow_request = asyncio.create_task(
                    client.post("/api/cameras", json={"name": "cam", "source": 0})
                )
                await asyncio.sleep(0.02)
                response = await client.get("/api/cameras")
                elapsed = time.perf_counter() - started
                await slow_request
                self.assertEqual(200, response.status_code)
                return elapsed

        self.assertLess(asyncio.run(exercise()), 0.15)


if __name__ == "__main__":
    unittest.main()
