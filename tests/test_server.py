"""
Automated Integration Tests for FastAPI Backend Server
Verifies:
- GET /api/health returns 200 and online status
- GET /api/samples returns 4 clinical samples
- POST /api/analyze with sample_id returns predictions, heatmaps, and metrics
- POST /api/analyze with raw file upload
"""

import io
import os
import sys
import unittest
from PIL import Image
from fastapi.testclient import TestClient

# Ensure repo root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from server import app


class TestServerAPI(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_health_endpoint(self):
        resp = self.client.get("/api/health")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "online")
        self.assertIn("torch_version", data)
        self.assertIn("device", data)

    def test_samples_endpoint(self):
        resp = self.client.get("/api/samples")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("samples", data)
        self.assertEqual(len(data["samples"]), 4)

    def test_analyze_with_sample_id(self):
        resp = self.client.post(
            "/api/analyze",
            data={"sample_id": "brain_mri", "beta": 0.50, "model_preset": "nano"}
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "success")
        self.assertIn("prediction", data)
        self.assertIn("confidence_pct", data["prediction"])
        self.assertIn("explainability", data)
        self.assertIn("faithfulness_pct", data["explainability"])
        self.assertIn("images", data)
        self.assertIn("fused", data["images"])
        self.assertTrue(data["images"]["fused"].startswith("data:image/png;base64,"))

    def test_analyze_with_file_upload(self):
        # Create dummy 100x100 RGB image in memory
        img = Image.new("RGB", (100, 100), color=(120, 140, 160))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)

        resp = self.client.post(
            "/api/analyze",
            files={"file": ("test_scan.png", buf, "image/png")},
            data={"beta": 0.60, "model_preset": "nano"}
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "success")
        self.assertIn("prediction", data)
        self.assertIn("images", data)


if __name__ == "__main__":
    unittest.main()
