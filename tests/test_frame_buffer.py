import unittest
from unittest.mock import patch

import numpy as np

from core.frame_buffer import FrameBuffer


class FrameBufferSnapshotTests(unittest.TestCase):
    def test_snapshot_separates_capture_publish_and_ai_metrics(self) -> None:
        buffer = FrameBuffer()

        with patch("core.frame_buffer.monotonic", return_value=10.0):
            buffer.update(
                np.zeros((2, 2, 3), dtype=np.uint8),
                object_count=0,
                captured_at_monotonic=9.75,
                capture_fps=25.0,
                dropped_capture_frames=3,
            )
        with patch("core.frame_buffer.monotonic", side_effect=[20.0, 20.25]):
            buffer.set_ai_latency(100.0)
            buffer.set_ai_latency(110.0)

        snapshot = buffer.snapshot()
        self.assertEqual(25.0, snapshot.capture_fps)
        self.assertEqual(3, snapshot.dropped_capture_frames)
        self.assertEqual(250.0, snapshot.capture_to_publish_ms)
        self.assertEqual(4.0, snapshot.ai_fps)

    def test_snapshot_includes_ai_latency(self) -> None:
        buffer = FrameBuffer()

        buffer.set_ai_latency(123.4)

        self.assertEqual(123.4, buffer.snapshot().ai_latency_ms)


if __name__ == "__main__":
    unittest.main()
