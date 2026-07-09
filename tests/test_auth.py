from __future__ import annotations

import unittest

from fastapi.testclient import TestClient

from web.app import create_app


class _AlertManager:
    _running = True


class _Runtime:
    def __init__(self) -> None:
        self.settings = {
            "web": {
                "auth": {
                    "enabled": True,
                    "username": "admin",
                    "password": "secret",
                    "session_secret": "test-secret",
                    "session_max_age_seconds": 3600,
                }
            }
        }
        self.alert_manager = _AlertManager()

    async def start(self) -> None:
        return None

    async def stop(self) -> None:
        return None

    def list_cameras(self) -> list[dict]:
        return []


class AuthTests(unittest.TestCase):
    def test_dashboard_redirects_to_login_without_session(self) -> None:
        client = TestClient(create_app(_Runtime()), follow_redirects=False)

        response = client.get("/")

        self.assertEqual(303, response.status_code)
        self.assertTrue(response.headers["location"].startswith("/login?next=%2F"))

    def test_api_requires_authentication(self) -> None:
        client = TestClient(create_app(_Runtime()))

        response = client.get("/api/cameras")

        self.assertEqual(401, response.status_code)

    def test_login_sets_session_cookie_for_api(self) -> None:
        client = TestClient(create_app(_Runtime()), follow_redirects=False)

        response = client.post(
            "/login",
            data={"username": "admin", "password": "secret", "next": "/"},
        )
        cameras = client.get("/api/cameras")

        self.assertEqual(303, response.status_code)
        self.assertIn("sct_camera_session", response.headers["set-cookie"])
        self.assertEqual(200, cameras.status_code)
        self.assertEqual([], cameras.json())

    def test_health_check_remains_public(self) -> None:
        client = TestClient(create_app(_Runtime()))

        response = client.get("/api/health")

        self.assertEqual(200, response.status_code)
        self.assertEqual("ok", response.json()["status"])


if __name__ == "__main__":
    unittest.main()
