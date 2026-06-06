"""HTTP client for the ROSE warm model server.

Usage::

    from rose.engine.server.warm_client import ROSEClient

    client = ROSEClient(port=5050)
    client.wait_ready(timeout_s=120)

    # 4DSG only (no VLM)
    result = client.build_4dsg("/path/to/video.mp4")
    print(result["four_dsg_dict"])

    # Full pipeline with VLM answer
    result = client.process_video("/path/to/video.mp4", "What is in front?")
    print(result["answer"])
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import requests

logger = logging.getLogger(__name__)


class ROSEClient:
    """Client for the ROSE warm server."""

    def __init__(self, host: str = "localhost", port: int = 5050):
        self._base = f"http://{host}:{port}"
        self._session = requests.Session()

    def status(self) -> dict:
        """Query server status."""
        resp = self._session.get(f"{self._base}/status", timeout=5)
        resp.raise_for_status()
        return resp.json()

    def wait_ready(self, timeout_s: float = 180, poll_interval_s: float = 2.0) -> bool:
        """Block until the server reports 'ready' status.

        Returns True if ready, False if timeout.
        """
        t0 = time.time()
        while time.time() - t0 < timeout_s:
            try:
                s = self.status()
                if s.get("status") == "ready":
                    return True
                logger.info(
                    "Server status: %s (%.0fs elapsed)",
                    s.get("status"), time.time() - t0,
                )
            except requests.ConnectionError:
                logger.debug("Server not yet reachable...")
            except Exception as e:
                logger.debug("Status check error: %s", e)
            time.sleep(poll_interval_s)
        return False

    def process_video(self, video_path: str, question: str) -> dict:
        """Full pipeline: video -> 4DSG -> VLM answer.

        Returns the full InferenceResponse as a dict with keys:
        status, answer, four_dsg_dict, scene_json, keyframe_dir,
        inference_time_s.
        """
        resp = self._session.post(
            f"{self._base}/infer",
            json={"video_path": str(video_path), "question": question},
            timeout=600,
        )
        resp.raise_for_status()
        return resp.json()

    def build_4dsg(self, video_path: str) -> dict:
        """4DSG only, no VLM query.

        Returns the full InferenceResponse as a dict.
        """
        resp = self._session.post(
            f"{self._base}/infer",
            json={"video_path": str(video_path)},
            timeout=600,
        )
        resp.raise_for_status()
        return resp.json()

    def shutdown(self) -> None:
        """Request graceful server shutdown."""
        try:
            self._session.post(f"{self._base}/shutdown", timeout=5)
        except Exception:
            pass  # Server may close connection before responding
