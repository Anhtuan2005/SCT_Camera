import unittest

from core.pipeline import CameraPipeline


class PipelineCadenceTests(unittest.TestCase):
    def test_analysis_due_respects_frame_skip_and_ai_max_fps(self) -> None:
        self.assertTrue(CameraPipeline._analysis_due(1, 1, 10, 100.0, 0.0))
        self.assertFalse(CameraPipeline._analysis_due(2, 3, 10, 100.2, 100.0))
        self.assertFalse(CameraPipeline._analysis_due(3, 3, 10, 100.05, 100.0))
        self.assertTrue(CameraPipeline._analysis_due(3, 3, 10, 100.11, 100.0))

    def test_analysis_due_allows_unlimited_ai_rate(self) -> None:
        self.assertTrue(CameraPipeline._analysis_due(5, 1, 0, 100.01, 100.0))

    def test_pose_needed_only_for_enabled_theft_asset_zones(self) -> None:
        settings = {
            "pose": {"enabled": True},
            "behavior": {"theft": {"enabled": True}},
        }
        asset_zone = {
            "type": "asset_watch",
            "polygon": [[0, 0], [1, 0], [1, 1]],
        }

        self.assertFalse(CameraPipeline._pose_needed({"zones": []}, settings))
        self.assertTrue(CameraPipeline._pose_needed({"zones": [asset_zone]}, settings))
        self.assertFalse(
            CameraPipeline._pose_needed(
                {"zones": [asset_zone]},
                {"pose": {"enabled": False}, "behavior": {"theft": {"enabled": True}}},
            )
        )
        self.assertFalse(
            CameraPipeline._pose_needed(
                {"zones": [asset_zone]},
                {"pose": {"enabled": True}, "behavior": {"theft": {"enabled": False}}},
            )
        )


if __name__ == "__main__":
    unittest.main()
