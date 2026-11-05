"""No camera/robot required: selection, calibration guards, pixel conversion and SDK lifecycle."""
import argparse
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from algorithm.camera_stream import (CameraSettings, OrbbecColorStream, add_camera_arguments,
    resolve_camera_arguments, validate_calibration, orbbec_frame_to_bgr, create_camera, paired_camera_settings)


FORMATS = types.SimpleNamespace(**{v: v for v in
    ("BGR", "RGB", "MJPG", "YUYV", "UYVY", "NV12", "NV21", "I420")})


class CameraTests(unittest.TestCase):
    def args(self, argv=()):
        parser = argparse.ArgumentParser()
        add_camera_arguments(parser)
        parser.add_argument("--intrinsics", default=None)
        parser.add_argument("--hand-eye", default=None)
        return parser.parse_args(argv)

    def test_terminal_selection_and_separate_paths(self):
        args = self.args()
        args.camera = None  # Legacy interactive API remains available.
        with patch("sys.stdin.isatty", return_value=True), \
                patch("builtins.input", side_effect=["bad", "2"]), \
                contextlib.redirect_stdout(io.StringIO()):
            settings = resolve_camera_arguments(args)
        self.assertEqual(settings.backend, "orbbec")
        self.assertIn("_orbbec_gemini305.json", args.intrinsics)
        self.assertIn("_orbbec_gemini305.json", args.hand_eye)

    def test_default_pair_and_eye_to_hand_paths(self):
        args = self.args()
        settings = resolve_camera_arguments(args)
        self.assertEqual((settings.backend, settings.model), ("orbbec", "gemini305"))
        secondary = paired_camera_settings(args)
        self.assertEqual(secondary.model, "gemini335")
        args = self.args()
        args.mode = "eye_to_hand"
        settings = resolve_camera_arguments(args)
        self.assertEqual(settings.model, "gemini335")
        self.assertIn("gemini335.json", args.intrinsics)
        self.assertIn("gemini335.json", args.hand_eye)
        args.single_camera = True
        self.assertIsNone(paired_camera_settings(args))

    def test_reject_other_orbbec_model_and_mount(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "calibration.json"
            data = {"camera": {"backend": "orbbec", "model": "Orbbec Gemini 335"}}
            path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "型号"):
                validate_calibration(path, CameraSettings("orbbec", model="gemini305"))
            data["mode"] = "eye_in_hand"
            path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "安装方式"):
                validate_calibration(path, CameraSettings("orbbec", model="gemini335"))

    def test_335_is_selected_by_model_even_when_305_is_first(self):
        ob, _ = self.sdk()
        device305 = ob.Context.return_value.query_devices.return_value.get_device_by_index.return_value
        device335 = Mock()
        info = device335.get_device_info.return_value
        info.get_name.return_value = "Orbbec Gemini 335"
        info.get_serial_number.return_value = "B"
        info.get_firmware_version.return_value = "1.4.60"
        devices = ob.Context.return_value.query_devices.return_value
        devices.get_count.return_value = 2
        devices.get_device_by_index.side_effect = [device305, device335]
        with patch.dict(sys.modules, pyorbbecsdk=ob), contextlib.redirect_stdout(io.StringIO()):
            stream = OrbbecColorStream(model="gemini335")
            try:
                ob.Pipeline.assert_called_once_with(device335)
                self.assertEqual(stream.metadata["role"], "eye_to_hand")
                self.assertEqual(stream.metadata["serial"], "B")
            finally:
                stream.stop()

    def test_optional_hand_eye_and_explicit_paths(self):
        args = self.args(["--camera", "orbbec", "--intrinsics", "custom.json"])
        args.optional_hand_eye = True
        with contextlib.redirect_stdout(io.StringIO()):
            resolve_camera_arguments(args)
        self.assertEqual(args.intrinsics, "custom.json")
        self.assertIsNone(args.hand_eye)

    def test_calibration_guards(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "calibration.json"
            def save(data):
                path.write_text(json.dumps(data))
            save({"image_size": [640, 480]})
            with contextlib.redirect_stdout(io.StringIO()):
                validate_calibration(path, CameraSettings())
            with self.assertRaisesRegex(ValueError, "元数据"):
                validate_calibration(path, CameraSettings("orbbec"))
            data = {"image_size": [640, 480], "camera": {"backend": "orbbec", "serial": "A"}}
            save(data)
            validate_calibration(path, CameraSettings("orbbec"), "A")
            with self.assertRaisesRegex(ValueError, "序列号"):
                validate_calibration(path, CameraSettings("orbbec"), "B")
            with self.assertRaisesRegex(ValueError, "分辨率"):
                validate_calibration(path, CameraSettings("orbbec", width=1280))
            with self.assertRaisesRegex(ValueError, "类型"):
                validate_calibration(path, CameraSettings())

    def frame(self, data, fmt, width=4, height=2):
        return types.SimpleNamespace(get_data=lambda: data, get_format=lambda: fmt,
                                     get_width=lambda: width, get_height=lambda: height)

    def test_color_order_ownership_and_mjpeg(self):
        bgr = np.full((2, 4, 3), (15, 90, 200), np.uint8)
        for fmt, data in (("BGR", bgr), ("RGB", bgr[:, :, ::-1].copy())):
            result = orbbec_frame_to_bgr(self.frame(data, fmt), FORMATS)
            np.testing.assert_array_equal(result, bgr)
            self.assertFalse(np.shares_memory(result, data))
        ok, jpg = cv2.imencode(".jpg", bgr)
        self.assertTrue(ok)
        result = orbbec_frame_to_bgr(self.frame(jpg, "MJPG"), FORMATS)
        self.assertLess(np.abs(result.astype(int) - bgr).max(), 5)
        with self.assertRaisesRegex(RuntimeError, "解码"):
            orbbec_frame_to_bgr(self.frame(b"invalid", "MJPG"), FORMATS)
        with self.assertRaisesRegex(RuntimeError, "尺寸"):
            orbbec_frame_to_bgr(self.frame(jpg, "MJPG", width=8), FORMATS)

    def test_yuv_formats(self):
        # Neutral chroma, studio-range black -> BGR black in every supported layout.
        for fmt in ("YUYV", "UYVY", "NV12", "NV21", "I420"):
            if fmt == "YUYV":
                raw = np.tile(np.array([16, 128], np.uint8), 8)
            elif fmt == "UYVY":
                raw = np.tile(np.array([128, 16], np.uint8), 8)
            else:
                raw = np.array([16] * 8 + [128] * 4, np.uint8)
            np.testing.assert_array_equal(orbbec_frame_to_bgr(self.frame(raw, fmt), FORMATS),
                                          np.zeros((2, 4, 3), np.uint8))

    def sdk(self):
        info = Mock()
        info.get_name.return_value = "Orbbec Gemini 305"
        info.get_serial_number.return_value = "A"
        info.get_firmware_version.return_value = "1.0.70"
        device = Mock()
        device.get_device_info.return_value = info
        devices = Mock()
        devices.get_count.return_value = 1
        devices.get_device_by_index.return_value = device
        profile = Mock()
        profile.as_video_stream_profile.return_value = profile
        profile.get_width.return_value, profile.get_height.return_value = 640, 480
        profile.get_fps.return_value, profile.get_format.return_value = 30, "MJPG"
        profiles = Mock()
        profiles.get_count.return_value = 1
        profiles.get_stream_profile_by_index.return_value = profile
        pipeline = Mock()
        pipeline.get_stream_profile_list.return_value = profiles
        ob = types.SimpleNamespace(Context=Mock(), Pipeline=Mock(return_value=pipeline),
            Config=Mock(), OBFormat=FORMATS,
            OBSensorType=types.SimpleNamespace(COLOR_SENSOR=1))
        ob.Context.return_value.query_devices.return_value = devices
        return ob, pipeline

    def test_exact_profile_timeout_and_stop(self):
        ob, pipeline = self.sdk()
        with patch.dict(sys.modules, pyorbbecsdk=ob), contextlib.redirect_stdout(io.StringIO()):
            stream = OrbbecColorStream()
            pipeline.start.assert_called_once()
            pipeline.wait_for_frames.return_value = None
            with patch("algorithm.camera_stream.time.monotonic", return_value=stream._last_frame + 6):
                with self.assertRaisesRegex(RuntimeError, "5 秒"):
                    stream.get_frame()
            stream.stop()
            stream.stop()
            pipeline.stop.assert_called_once()
            with self.assertRaisesRegex(RuntimeError, "可用彩色流"):
                OrbbecColorStream(width=1280)

    def test_factory_stops_camera_on_serial_mismatch(self):
        ob, pipeline = self.sdk()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "intr.json"
            path.write_text(json.dumps({"camera": {"backend": "orbbec", "serial": "B"}}))
            with patch.dict(sys.modules, pyorbbecsdk=ob), contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(ValueError, "序列号"):
                    create_camera(CameraSettings("orbbec"), [path])
            pipeline.stop.assert_called_once()

    def test_factory_intrinsics_use_opencv_rational_order(self):
        ob, _ = self.sdk()
        profile = ob.Pipeline.return_value.get_stream_profile_list.return_value.get_stream_profile_by_index.return_value
        profile.get_intrinsic.return_value = types.SimpleNamespace(
            width=640, height=480, fx=370., fy=371., cx=316., cy=239.)
        profile.get_distortion.return_value = types.SimpleNamespace(
            k1=1., k2=2., k3=3., k4=4., k5=5., k6=6., p1=.1, p2=.2)
        with patch.dict(sys.modules, pyorbbecsdk=ob), contextlib.redirect_stdout(io.StringIO()):
            stream = OrbbecColorStream()
            try:
                intr = stream.get_factory_intrinsics()
            finally:
                stream.stop()
        self.assertEqual(intr["dist_coeffs"], [1., 2., .1, .2, 3., 4., 5., 6.])
        self.assertEqual(intr["camera_matrix"][0], [370., 0., 316.])
        self.assertEqual(intr["image_size"], [640, 480])
        self.assertEqual(intr["source"], "factory_sdk")


if __name__ == "__main__":
    unittest.main()
