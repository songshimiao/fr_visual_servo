"""Color-only camera adapters. Images are owned uint8 BGR arrays, never resized.

Camera optical frame: x right, y down, z forward; PnP translation remains metres.
Vendor SDKs are imported only after camera selection, so either SDK is optional.
"""
import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np


@dataclass(frozen=True)
class CameraSettings:
    backend: str = "realsense"
    width: int = 640
    height: int = 480
    fps: int = 30
    serial: str = None
    model: str = None


def positive_int(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("必须为正整数")
    return value


def add_camera_arguments(parser):
    parser.add_argument("--camera", choices=("realsense", "orbbec"), default="orbbec",
                        help="默认 Orbbec 双相机：305 eye-in-hand / 335 eye-to-hand")
    parser.add_argument("--camera-model", choices=("gemini305", "gemini335"), default=None,
                        help="主相机型号；默认 gemini305，eye_to_hand 标定默认 gemini335")
    parser.add_argument("--single-camera", action="store_true", help="仅打开所选主相机")
    parser.add_argument("--secondary-camera-serial", default=None, help="另一台相机序列号")
    parser.add_argument("--camera-width", type=positive_int, default=640)
    parser.add_argument("--camera-height", type=positive_int, default=480)
    parser.add_argument("--camera-fps", type=positive_int, default=30)
    parser.add_argument("--camera-serial", default=None, help="指定相机序列号")


def resolve_camera_arguments(args):
    """Run on the main thread, before connecting a robot or starting workers."""
    if args.camera is None:
        if not sys.stdin.isatty():
            raise ValueError("非交互终端请指定 --camera realsense 或 --camera orbbec")
        print("选择相机：\n  1. Intel RealSense D435\n  2. Orbbec Gemini 305")
        while True:
            choice = input("请输入 1/2（q 退出）: ").strip().lower()
            if choice in ("1", "realsense", "2", "orbbec"):
                args.camera = "realsense" if choice in ("1", "realsense") else "orbbec"
                break
            if choice == "q":
                raise SystemExit(0)
            print("无效选择，请输入 1 或 2")
    model = getattr(args, "camera_model", None)
    if args.camera == "orbbec":
        model = model or ("gemini335" if getattr(args, "mode", None) == "eye_to_hand" else "gemini305")
        args.camera_model = model
        expected_mode = camera_role(model)
        if getattr(args, "cmd", None) == "extrinsic" and getattr(args, "mode", None) not in (None, expected_mode):
            raise ValueError(f"{model} 的安装方式固定为 {expected_mode}，与 --mode 不一致")
    suffix = f"_orbbec_{model}" if args.camera == "orbbec" else ""
    defaults = {"intrinsics": f"resources/calibration/camera_intrinsics{suffix}.json",
                "hand_eye": f"resources/calibration/hand_eye_calibration{suffix}.json",
                "out": (f"resources/calibration/camera_intrinsics{suffix}.json"
                        if getattr(args, "cmd", None) == "intrinsic" else
                        f"resources/calibration/hand_eye_calibration{suffix}.json")}
    for name, path in defaults.items():
        if hasattr(args, name) and getattr(args, name) is None:
            # ArUco's hand-eye is explicitly optional; do not connect a robot by default.
            if name == "hand_eye" and getattr(args, "optional_hand_eye", False):
                continue
            setattr(args, name, path)
    settings = CameraSettings(args.camera, args.camera_width, args.camera_height,
                              args.camera_fps, args.camera_serial, model)
    print(f"[相机] {settings.backend} {settings.width}x{settings.height}@{settings.fps}")
    return settings


def camera_role(model):
    return {"gemini305": "eye_in_hand", "gemini335": "eye_to_hand"}[model]


def camera_label(model):
    return {"gemini305": "Gemini 305 · Eye-in-Hand（眼在手上）",
            "gemini335": "Gemini 335 · Eye-to-Hand（眼在手外）"}[model]


def paired_camera_settings(args):
    """Primary settings must already have been resolved on the main thread."""
    if args.camera != "orbbec" or getattr(args, "single_camera", False):
        return None
    model = "gemini335" if args.camera_model == "gemini305" else "gemini305"
    return CameraSettings("orbbec", args.camera_width, args.camera_height,
                          args.camera_fps, args.secondary_camera_serial, model)


def calibration_path(model, kind="camera_intrinsics"):
    return f"resources/calibration/{kind}_orbbec_{model}.json"


def validate_calibration(path, settings, serial=None):
    """Keep legacy RealSense JSON usable; new Orbbec files must identify the camera."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    metadata = data.get("camera")
    if metadata is None:
        if settings.backend != "realsense":
            raise ValueError(f"{path}: 缺少 camera 元数据，请为当前 Gemini 相机重新标定，不能复用 RealSense 参数")
        print(f"[标定] {path}: 旧版 RealSense 文件无相机标识，请确认相机和安装位置一致")
    else:
        if metadata["backend"] != settings.backend:
            raise ValueError(f"{path}: 标定相机类型与所选相机不一致")
        if settings.backend == "orbbec" and settings.model:
            expected_model = settings.model.removeprefix("gemini")
            if expected_model not in metadata.get("model", ""):
                raise ValueError(f"{path}: 标定相机型号与 {settings.model} 不一致")
            role = metadata.get("role")
            if role is not None and role != camera_role(settings.model):
                raise ValueError(f"{path}: 标定相机安装方式不一致")
            if data.get("mode") not in (None, camera_role(settings.model)):
                raise ValueError(f"{path}: 手眼标定 mode 与相机安装方式不一致")
        expected_serial = serial or settings.serial
        if expected_serial and metadata.get("serial") != expected_serial:
            raise ValueError(f"{path}: 标定相机序列号与当前相机不一致")
    size = data.get("image_size")
    if size and list(size) != [settings.width, settings.height]:
        raise ValueError(f"{path}: 内参分辨率 {size} 与采集分辨率不一致，请重新标定")
    return data


class RealSenseColorStream:
    def __init__(self, width=640, height=480, fps=30, serial=None):
        try:
            import pyrealsense2 as rs
        except ImportError as exc:
            raise RuntimeError("缺少 pyrealsense2，请使用原 RealSense Python 环境") from exc
        self.pipeline = rs.pipeline()
        cfg = rs.config()
        if serial:
            cfg.enable_device(serial)
        cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        self.profile = self.pipeline.start(cfg)
        self._running = True
        self._last_frame = time.monotonic()
        self.metadata = {"backend": "realsense", "serial":
                         self.profile.get_device().get_info(rs.camera_info.serial_number),
                         "width": width, "height": height, "fps": fps}

    def get_frame(self):
        color = self.pipeline.wait_for_frames(timeout_ms=2000).get_color_frame()
        if not color:
            if time.monotonic() - self._last_frame > 5:
                raise RuntimeError("RealSense 彩色流连续 5 秒无有效帧")
            return None
        self._last_frame = time.monotonic()
        return np.asanyarray(color.get_data()).copy()

    def get_color_frame(self):
        return self.get_frame()

    def get_rs_intrinsics(self):
        import pyrealsense2 as rs
        intr = self.profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        return dict(width=intr.width, height=intr.height, fx=intr.fx, fy=intr.fy,
                    cx=intr.ppx, cy=intr.ppy, dist_coeffs=list(intr.coeffs), model=str(intr.model))

    def get_factory_intrinsics(self):
        intr = self.get_rs_intrinsics()
        if "inverse_brown" in intr["model"] or "kannala" in intr["model"]:
            raise ValueError("该 RealSense 畸变模型需专门转换，请使用 ChArUco 标定")
        return _intrinsics_json(intr, self.metadata)

    def stop(self):
        if self._running:
            self._running = False
            self.pipeline.stop()


def orbbec_frame_to_bgr(frame, formats):
    w, h, fmt = frame.get_width(), frame.get_height(), frame.get_format()
    data = np.frombuffer(frame.get_data(), dtype=np.uint8)
    if fmt == formats.MJPG:
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError("Orbbec MJPEG 解码失败")
    elif fmt in (formats.RGB, formats.BGR):
        image = data.reshape(h, w, 3)
        if fmt == formats.RGB:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    elif fmt in (formats.YUYV, formats.UYVY):
        code = cv2.COLOR_YUV2BGR_YUY2 if fmt == formats.YUYV else cv2.COLOR_YUV2BGR_UYVY
        image = cv2.cvtColor(data.reshape(h, w, 2), code)
    elif fmt in (formats.NV12, formats.NV21, formats.I420):
        code = {formats.NV12: cv2.COLOR_YUV2BGR_NV12,
                formats.NV21: cv2.COLOR_YUV2BGR_NV21,
                formats.I420: cv2.COLOR_YUV2BGR_I420}[fmt]
        image = cv2.cvtColor(data.reshape(h * 3 // 2, w), code)
    else:
        raise RuntimeError(f"不支持的 Orbbec 彩色格式: {fmt}")
    if image.shape != (h, w, 3):
        raise RuntimeError("Orbbec 彩色帧尺寸不符合流配置")
    return image.copy()


class OrbbecColorStream:
    def __init__(self, width=640, height=480, fps=30, serial=None, model="gemini305"):
        try:
            import pyorbbecsdk as ob
        except ImportError as exc:
            raise RuntimeError("缺少 Orbbec SDK v2，请运行 bash scripts/setup_orbbec_jetson.sh") from exc
        model_number = model.removeprefix("gemini")
        self.model = model
        self.ob = ob
        self.context = ob.Context()
        devices = self.context.query_devices()
        matches = []
        for index in range(devices.get_count()):
            device = devices.get_device_by_index(index)
            info = device.get_device_info()
            if model_number in info.get_name() and (serial is None or info.get_serial_number() == serial):
                matches.append(device)
        if len(matches) != 1:
            raise RuntimeError(f"找到 {len(matches)} 台匹配的 Gemini {model_number}，请检查 USB/udev；多台时用 --camera-serial")
        self.device = matches[0]
        info = self.device.get_device_info()
        self.pipeline = ob.Pipeline(self.device)
        profiles = self.pipeline.get_stream_profile_list(ob.OBSensorType.COLOR_SENSOR)
        available = [profiles.get_stream_profile_by_index(i).as_video_stream_profile()
                     for i in range(profiles.get_count())]
        # Choose format only, never silently change calibrated geometry or frame rate.
        profile = None
        for fmt in (ob.OBFormat.BGR, ob.OBFormat.RGB, ob.OBFormat.MJPG,
                    ob.OBFormat.YUYV, ob.OBFormat.UYVY, ob.OBFormat.NV12,
                    ob.OBFormat.NV21, ob.OBFormat.I420):
            profile = next((p for p in available if p.get_width() == width and
                            p.get_height() == height and p.get_fps() == fps and
                            p.get_format() == fmt), None)
            if profile is not None:
                break
        if profile is None:
            options = ", ".join(f"{p.get_width()}x{p.get_height()}@{p.get_fps()} {p.get_format()}"
                                for p in available)
            raise RuntimeError(f"Gemini {model_number} 不支持 {width}x{height}@{fps}；可用彩色流: {options}")
        cfg = ob.Config()
        cfg.enable_stream(profile)
        self.color_profile = profile
        self.pipeline.start(cfg)
        self._running = True
        self._last_frame = time.monotonic()
        self.metadata = dict(backend="orbbec", model=info.get_name(),
                             serial=info.get_serial_number(), role=camera_role(model), firmware=info.get_firmware_version(),
                             width=width, height=height, fps=fps, format=str(profile.get_format()))
        print(f"[Orbbec] {self.metadata}")

    def get_frame(self):
        frames = self.pipeline.wait_for_frames(1000)
        color = frames.get_color_frame() if frames else None
        if not color:
            if time.monotonic() - self._last_frame > 5:
                raise RuntimeError(f"{self.model} 彩色流连续 5 秒无有效帧，请检查 USB 连接")
            return None
        image = orbbec_frame_to_bgr(color, self.ob.OBFormat)
        self._last_frame = time.monotonic()
        return image

    def get_color_frame(self):
        return self.get_frame()

    def get_factory_intrinsics(self):
        intr = self.color_profile.get_intrinsic()
        dist = self.color_profile.get_distortion()
        # OpenCV rational Brown order differs from the SDK struct field order.
        values = dict(width=intr.width, height=intr.height, fx=intr.fx, fy=intr.fy,
                      cx=intr.cx, cy=intr.cy,
                      dist_coeffs=[dist.k1, dist.k2, dist.p1, dist.p2,
                                   dist.k3, dist.k4, dist.k5, dist.k6])
        return _intrinsics_json(values, self.metadata)

    def stop(self):
        if self._running:
            self._running = False
            self.pipeline.stop()


def create_camera(settings=None, calibration_paths=()):
    settings = settings or CameraSettings()
    # Fail before opening hardware when calibration belongs to another backend/size.
    for path in calibration_paths:
        validate_calibration(path, settings)
    cls = {"realsense": RealSenseColorStream, "orbbec": OrbbecColorStream}[settings.backend]
    kwargs = {"model": settings.model or "gemini305"} if settings.backend == "orbbec" else {}
    stream = cls(settings.width, settings.height, settings.fps, settings.serial, **kwargs)
    try:
        for path in calibration_paths:
            validate_calibration(path, settings, stream.metadata["serial"])
    except Exception:
        stream.stop()
        raise
    return stream


def _intrinsics_json(intr, metadata):
    return {"source": "factory_sdk", "camera": dict(metadata),
            "image_size": [intr["width"], intr["height"]],
            "camera_matrix": [[intr["fx"], 0., intr["cx"]],
                              [0., intr["fy"], intr["cy"]], [0., 0., 1.]],
            "dist_coeffs": intr["dist_coeffs"],
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}
