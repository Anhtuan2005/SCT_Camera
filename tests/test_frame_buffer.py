import unittest

from core.frame_buffer import FrameBuffer


class FrameBufferSnapshotTests(unittest.TestCase):
    def test_snapshot_includes_ai_latency(self) -> None:
        buffer = FrameBuffer()

        buffer.set_ai_latency(123.4)

        self.assertEqual(123.4, buffer.snapshot().ai_latency_ms)


if __name__ == "__main__":
    unittest.main()
