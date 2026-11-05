"""Offline tests: real HTTP/JPEG, synthetic frames, and isolated camera call sites.

Run from the repository root: python3 -m unittest discover -s tests -v
No RealSense pipeline or robot SDK is started.
"""

import ast
import argparse
import contextlib
import http.client
import io
import json
from pathlib import Path
import sys
import threading
import time
import types
import unittest
from unittest.mock import Mock, patch
import __future__

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from algorithm.web_preview import WebPreview  # noqa: F401


def wait_until(predicate, timeout=3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("Timed out waiting for preview")


def isolated_definition(path, name, namespace):
    """Exercise the actual function without importing camera/robot dependencies."""
    tree = ast.parse((ROOT / path).read_text())
    node = next(n for n in tree.body if getattr(n, "name", None) == name)
    code = compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec",
                   flags=__future__.annotations.compiler_flag)
    exec(code, namespace)
    return namespace[name]


class PreviewTests(unittest.TestCase):
    def setUp(self):
        self.preview = WebPreview(host="127.0.0.1", port=0, fps=60,
                                  allow_capture=True, title="Test <camera>").start()

    def tearDown(self):
        self.preview.close()

    def request(self, path, method="GET", token=True):
        conn = http.client.HTTPConnection("127.0.0.1", self.preview.port, timeout=3)
        headers = {"X-Preview-Token": self.preview._token} if token else {}
        conn.request(method, path, headers=headers)
        response = conn.getresponse()
        result = response.status, response.read()
        conn.close()
        return result

    def test_page_status_and_real_jpeg(self):
        code, page = self.request("/")
        self.assertEqual(code, 200)
        self.assertIn(b"Test &lt;camera&gt;", page)
        self.assertIsNone(json.loads(self.request("/status")[1])["frame_age_s"])
        frame = np.full((48, 64, 3), (20, 80, 180), dtype=np.uint8)
        self.preview.publish(frame)
        wait_until(lambda: self.preview.status()["frame_sequence"] == 1)
        # Two browsers read the same encoded frame; disconnecting one does not quit.
        for _ in range(2):
            conn = http.client.HTTPConnection("127.0.0.1", self.preview.port, timeout=3)
            conn.request("GET", "/stream.mjpg")
            response = conn.getresponse()
            self.assertEqual(response.status, 200)
            self.assertIn("multipart/x-mixed-replace", response.getheader("Content-Type"))
            self.assertEqual(response.readline(), b"--frame\r\n")
            self.assertEqual(response.readline(), b"Content-Type: image/jpeg\r\n")
            length = int(response.readline().split(b":")[1])
            self.assertEqual(response.readline(), b"\r\n")
            jpeg = response.read(length)
            decoded = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
            self.assertEqual(decoded.shape, frame.shape)
            conn.close()
        self.assertFalse(self.preview.stop_event.is_set())

    def test_two_real_streams_and_independent_save_requests(self):
        self.preview.close()
        self.preview = WebPreview(host="127.0.0.1", port=0, fps=60, allow_capture=True)
        secondary = self.preview.add_channel("secondary", "Gemini 335 / Eye-to-Hand")
        self.preview.start()
        frames = {"primary": np.full((24, 32, 3), (0, 0, 220), np.uint8),
                  "secondary": np.full((24, 32, 3), (0, 220, 0), np.uint8)}
        for name, frame in frames.items():
            self.preview.channels[name].publish(frame)
        wait_until(lambda: all(c.status()["frame_sequence"] for c in self.preview.channels.values()))
        page = self.request("/")[1]
        self.assertEqual(page.count(b"<button data-capture="), 2)
        for name, frame in frames.items():
            self.assertIn(f'/stream/{name}.mjpg'.encode(), page)
            conn = http.client.HTTPConnection("127.0.0.1", self.preview.port, timeout=3)
            conn.request("GET", f"/stream/{name}.mjpg")
            response = conn.getresponse()
            self.assertEqual(response.status, 200)
            response.readline()
            response.readline()
            length = int(response.readline().split(b":")[1])
            response.readline()
            decoded = cv2.imdecode(np.frombuffer(response.read(length), np.uint8), cv2.IMREAD_COLOR)
            np.testing.assert_allclose(decoded, frame, atol=3)
            conn.close()
        self.assertEqual(self.request("/command/sample/secondary", "POST")[0], 200)
        self.assertEqual(self.preview.poll_key(), -1)
        self.assertEqual(secondary.poll_key(), ord("s"))
        self.assertEqual(secondary.poll_key(), -1)
        self.assertEqual(self.request("/command/sample/primary", "POST")[0], 200)
        self.assertEqual(self.preview.poll_key(), ord("s"))
        self.assertEqual(secondary.poll_key(), -1)
        self.assertEqual(self.request("/stream/unknown.mjpg")[0], 404)
        self.assertEqual(self.request("/command/sample/unknown", "POST")[0], 404)
        self.assertEqual(self.request("/command/sample/secondary", "POST", token=False)[0], 403)
        secondary.set_feedback(saved=3, target=20, message="sample saved")
        state = json.loads(self.request("/status")[1])["channels"]
        self.assertEqual(state["secondary"]["saved"], 3)
        self.assertEqual(state["primary"]["saved"], 0)
        secondary.allow_capture = self.preview.allow_capture = False
        self.assertNotIn(b'data-capture="primary"', self.request("/")[1])
        self.assertNotIn(b'data-capture="secondary"', self.request("/")[1])
        self.assertEqual(self.request("/command/sample/secondary", "POST")[0], 404)
        self.request("/command/quit", "POST")
        self.assertEqual(secondary.poll_key(), ord("q"))

    def test_command_guards_and_latched_quit(self):
        self.assertEqual(self.request("/command/quit")[0], 404)
        self.assertEqual(self.request("/command/quit", "POST", token=False)[0], 403)
        self.assertFalse(self.preview.stop_event.is_set())
        self.assertEqual(self.request("/command/sample", "POST")[0], 200)
        self.assertEqual(self.preview.poll_key(), ord("s"))
        self.assertEqual(self.preview.poll_key(), -1)
        self.assertEqual(self.request("/command/quit", "POST")[0], 200)
        self.assertEqual(self.preview.poll_key(), ord("q"))
        self.assertEqual(self.preview.poll_key(), ord("q"))

    def test_sample_is_disabled_for_monitor_only(self):
        self.preview.allow_capture = False
        self.assertEqual(self.request("/command/sample", "POST")[0], 404)
        self.assertEqual(self.request("/command/execute", "POST")[0], 404)

    def test_publish_is_not_blocked_by_encoder_and_keeps_latest(self):
        started, release = threading.Event(), threading.Event()
        real_encode = cv2.imencode

        def slow_encode(*args):
            started.set()
            release.wait(3)
            return real_encode(*args)

        frame = np.zeros((32, 32, 3), dtype=np.uint8)
        with patch("algorithm.web_preview.cv2.imencode", side_effect=slow_encode):
            self.preview.publish(frame)
            self.assertTrue(started.wait(2))
            try:
                time.sleep(0.02)
                self.preview.publish(frame + 30)
                time.sleep(0.02)
                self.preview.publish(frame + 60)
                with self.preview._condition:
                    self.assertEqual(int(self.preview._raw[0][0, 0, 0]), 60)
                    self.assertEqual(self.preview._sequence, 0)
            finally:
                release.set()
            wait_until(lambda: self.preview.status()["frame_sequence"] == 2)

    def test_encoding_failure_is_reported_and_recovers(self):
        frame = np.zeros((32, 32, 3), dtype=np.uint8)
        with patch("algorithm.web_preview.cv2.imencode", return_value=(False, None)):
            self.preview.publish(frame)
            wait_until(lambda: self.preview.status()["error"] is not None)
        time.sleep(0.02)
        self.preview.publish(frame)
        wait_until(lambda: self.preview.status()["frame_sequence"] == 1)
        self.assertIsNone(self.preview.status()["error"])

    def test_close_releases_workers_and_port(self):
        port = self.preview.port
        self.preview.close()
        self.assertFalse(self.preview._encoder_thread.is_alive())
        self.assertFalse(self.preview._http_thread.is_alive())
        with WebPreview(host="127.0.0.1", port=port):
            pass


class CameraIntegrationTests(unittest.TestCase):
    def test_hover_dry_run_cancel_and_detection_failure_never_move(self):
        class Pose:
            t = np.zeros(3)

            def __mul__(self, other):
                return self

            def rpy(self, **kwargs):
                return np.zeros(3)

        for scenario in ("dry-run", "cancel", "detection-failure", "web-quit"):
            with self.subTest(scenario=scenario):
                rpc = Mock()
                rpc.GetActualJointPosDegree.return_value = (0, [0] * 6)
                model = Mock()
                model.forward_kinematics.return_value = Pose()
                model.inverse_kinematics.return_value = np.zeros(6)
                stream = Mock()
                stream.get_frame.return_value = None
                preview = WebPreview(enabled=False)

                def reference(*args, **kwargs):
                    if scenario == "detection-failure":
                        raise RuntimeError("no marker")
                    if scenario == "web-quit":
                        preview.stop_event.set()
                    return Pose()

                sdk_module = types.ModuleType("robot.fairino")
                sdk_module.Robot = types.SimpleNamespace(RPC=Mock(return_value=rpc))
                model_module = types.ModuleType("robot.fairino_robot")
                model_module.FairinoRobot = Mock(return_value=model)
                env = {"argparse": argparse, "sys": sys, "np": np, "Path": Path,
                       "__file__": str(ROOT / "scripts/visual_servo_hover_test.py"),
                       "threading": threading, "time": time,
                       "add_web_arguments": lambda parser: None,
                       "add_camera_arguments": lambda parser: None,
                       "resolve_camera_arguments": lambda args: (setattr(args, "intrinsics", "intr.json"), setattr(args, "hand_eye", "hand.json")),
                       "validate_calibration": lambda *a: None,
                       "load_camera_intrinsics": lambda _: (None, None),
                       "load_hand_eye": lambda _: {"mode": "eye_in_hand", "T": None},
                       "np_to_SE3": lambda _: Pose(),
                       "WebPreview": types.SimpleNamespace(from_args=lambda *a, **kw: preview),
                       "create_camera": lambda *a: stream,
                       "wait_for_stable_marker": reference,
                       "compute_target_flange_pose": lambda *a: Pose()}
                main = isolated_definition("scripts/visual_servo_hover_test.py", "main", env)
                argv = ["hover", "--mode", "camera"]
                if scenario != "dry-run":
                    argv.append("--execute")
                with patch.dict(sys.modules, {"robot.fairino": sdk_module,
                                              "robot.fairino_robot": model_module}), \
                        patch.object(sys, "argv", argv), \
                        patch("builtins.input", return_value="no" if scenario == "cancel" else "yes"), \
                        contextlib.redirect_stdout(io.StringIO()):
                    if scenario == "detection-failure":
                        with self.assertRaisesRegex(RuntimeError, "no marker"):
                            main()
                    else:
                        main()
                rpc.RobotEnable.assert_not_called()
                rpc.MoveJ.assert_not_called()
                stream.stop.assert_called_once()

    def environment(self):
        frame = np.zeros((24, 32, 3), np.uint8)
        camera = types.SimpleNamespace(get_color_frame=lambda: frame)
        cv = types.SimpleNamespace(
            cvtColor=lambda frame, _: frame[:, :, 0], COLOR_BGR2GRAY=0,
            putText=Mock(), FONT_HERSHEY_SIMPLEX=0,
            aruco=types.SimpleNamespace(drawDetectedCornersCharuco=Mock()))
        env = {"cv2": cv, "np": np,
               "_detect_charuco": lambda *_: (np.zeros((4, 1, 2)), np.arange(4)),
               "estimate_target_pose_in_camera": lambda *_: np.eye(4)}
        return camera, env

    def test_web_sample_collects_intrinsics_and_hand_eye(self):
        camera, env = self.environment()
        intrinsic = isolated_definition("scripts/algorithm/camera_calibration.py",
                                        "collect_charuco_images", env)
        hand_eye = isolated_definition("scripts/algorithm/camera_calibration.py",
                                       "collect_hand_eye_samples", env)
        with contextlib.redirect_stdout(io.StringIO()):
            with WebPreview(enabled=False, allow_capture=True) as preview:
                preview._sample.set()
                corners, ids, size = intrinsic(camera, None, None, 1, preview=preview)
                self.assertEqual((len(corners), len(ids), size), (1, 1, (32, 24)))
            with WebPreview(enabled=False, allow_capture=True) as preview:
                preview._sample.set()
                pose_provider = types.SimpleNamespace(get_pose_matrix=lambda: np.eye(4))
                result = hand_eye(camera, None, None, None, None, pose_provider, 1,
                                  preview=preview)
                self.assertEqual([len(v) for v in result], [1, 1, 1, 1])

    def test_quit_before_camera_read(self):
        camera, env = self.environment()
        camera.get_color_frame = Mock(side_effect=AssertionError("camera must not be read"))
        intrinsic = isolated_definition("scripts/algorithm/camera_calibration.py",
                                        "collect_charuco_images", env)
        with WebPreview(enabled=False) as preview:
            preview.stop_event.set()
            with contextlib.redirect_stdout(io.StringIO()):
                result = intrinsic(camera, None, None, 20, preview=preview)
            self.assertEqual(result, ([], [], None))

    def test_vision_camera_failure_closes_preview_and_signals_stop(self):
        preview = WebPreview(host="127.0.0.1", port=0)
        env = {"threading": threading,
               "create_camera": Mock(side_effect=RuntimeError("camera absent"))}
        worker_class = isolated_definition("scripts/main_pbvs.py", "VisionWorker", env)
        stop = threading.Event()
        worker = worker_class(None, None, 0.05, "DICT_4X4_50", None, None, stop, preview)
        with self.assertRaisesRegex(RuntimeError, "camera absent"):
            worker.run()
        self.assertTrue(stop.is_set())
        self.assertFalse(preview._http_thread.is_alive())

    def test_no_highgui_calls_remain(self):
        for path in (ROOT / "scripts").rglob("*.py"):
            if "fairino" in path.parts:
                continue
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                    self.assertNotIn(node.func.attr, {"imshow", "waitKey", "destroyAllWindows"},
                                     f"HighGUI call in {path}:{node.lineno}")

    def test_disabled_preview_and_argument_validation(self):
        with WebPreview(enabled=False) as preview:
            self.assertIsNone(preview._server)
            preview.publish(None)
        for kwargs in ({"fps": 0}, {"quality": 101}, {"port": -1}):
            with self.assertRaises(ValueError):
                WebPreview(**kwargs)


if __name__ == "__main__":
    unittest.main()
