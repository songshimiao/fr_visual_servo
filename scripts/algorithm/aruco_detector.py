"""
scripts/algorithm/aruco_detector.py

RealSense D435 实时 ArUco 坐标系检测模块。
依赖 camera_calibration.py 产出的内参 (与可选的手眼标定结果)。

依赖:
    pip install opencv-contrib-python pyrealsense2 numpy

用法示例:
    # 仅显示 marker 在相机坐标系下的位姿
    python aruco_detector.py --intrinsics resources/calibration/camera_intrinsics.json \
        --marker-length 0.04 --dict DICT_4X4_50

    # 同时换算到机器人基坐标系 (会自动读取 hand-eye json 中记录的 mode)
    python aruco_detector.py --intrinsics resources/calibration/camera_intrinsics.json \
        --hand-eye resources/calibration/hand_eye_calibration.json \
        --marker-length 0.04
"""

import argparse
import json
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pyrealsense2 as rs


def _get_aruco_detector(dict_name: str):
    dict_id = getattr(cv2.aruco, dict_name)
    aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)
    if hasattr(cv2.aruco, "ArucoDetector"):
        params = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(aruco_dict, params)
        return detector, None
    else:
        print(f"[OpenCV] 版本过低, OpenCV >= 4.7")


class RealSenseColorStream:
    def __init__(self, width=640, height=480, fps=30):
        self.pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        self.pipeline.start(cfg)

    def get_frame(self) -> Optional[np.ndarray]:
        frames = self.pipeline.wait_for_frames()
        color = frames.get_color_frame()
        if not color:
            return None
        return np.asanyarray(color.get_data())

    def stop(self):
        self.pipeline.stop()


def load_camera_intrinsics(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    camera_matrix = np.array(data["camera_matrix"], dtype=np.float64)
    dist_coeffs = np.array(data["dist_coeffs"], dtype=np.float64)
    return camera_matrix, dist_coeffs


def load_hand_eye(path: Optional[Path]):
    """加载手眼标定结果，自动读取标定时选择的 mode (eye_in_hand / eye_to_hand)"""
    if path is None:
        return None
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    T = np.array(data["cam_T_gripper_or_base"], dtype=np.float64)
    return {"mode": data["mode"], "T": T}


def detect_markers(gray, dict_name: str):
    """检测 ArUco 标记，返回 (corners, ids)"""
    if hasattr(cv2.aruco, "ArucoDetector"):
        detector, _ = _get_aruco_detector(dict_name)
        corners, ids, _ = detector.detectMarkers(gray)
    else:
        _, (aruco_dict, params) = _get_aruco_detector(dict_name)
        corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=params)
    return corners, ids


def estimate_marker_poses(corners, marker_length, camera_matrix, dist_coeffs) -> list:
    """
    估计每个 marker 相对相机的位姿 (使用 solvePnP, 兼容各 OpenCV 版本，
    避免依赖已在新版本弃用的 estimatePoseSingleMarkers)。
    返回每个 marker 的 4x4 齐次矩阵 cam_T_marker 列表，顺序与 corners 对应。
    """
    half = marker_length / 2.0
    obj_points = np.array([
        [-half,  half, 0],
        [ half,  half, 0],
        [ half, -half, 0],
        [-half, -half, 0],
    ], dtype=np.float64)

    poses = []
    for c in corners:
        img_points = c.reshape(4, 2).astype(np.float64)
        ok, rvec, tvec = cv2.solvePnP(
            obj_points, img_points, camera_matrix, dist_coeffs,
            flags=cv2.SOLVEPNP_IPPE_SQUARE
        )
        if not ok:
            poses.append(None)
            continue
        R, _ = cv2.Rodrigues(rvec)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = tvec.flatten()
        poses.append(T)
    return poses


def transform_pose(cam_T_marker: np.ndarray, hand_eye: Optional[dict],
                    base_T_gripper: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
    """
    将 marker 在相机坐标系下的位姿换算到机器人基坐标系。

    - eye_to_hand: 已知 cam_T_base, 直接 base_T_marker = inv(cam_T_base) @ cam_T_marker
    - eye_in_hand: 已知 cam_T_gripper, 还需当前 base_T_gripper (实时读取机械臂位姿)
                   base_T_marker = base_T_gripper @ cam_T_gripper @ cam_T_marker
    """
    if hand_eye is None:
        return None

    if hand_eye["mode"] == "eye_to_hand":
        cam_T_base = hand_eye["T"]
        base_T_cam = np.linalg.inv(cam_T_base)
        return base_T_cam @ cam_T_marker

    elif hand_eye["mode"] == "eye_in_hand":
        if base_T_gripper is None:
            raise ValueError("eye_in_hand 模式需要传入实时 base_T_gripper")
        cam_T_gripper = hand_eye["T"]
        return base_T_gripper @ cam_T_gripper @ cam_T_marker

    return None


def draw_axis(frame, camera_matrix, dist_coeffs, T: np.ndarray, length=0.03):
    rvec, _ = cv2.Rodrigues(T[:3, :3])
    tvec = T[:3, 3]
    cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, length)


def run_stream(camera_matrix, dist_coeffs, marker_length, dict_name,
                hand_eye: Optional[dict] = None, pose_provider=None):
    stream = RealSenseColorStream()
    print("[运行] 按 'q' 退出")
    try:
        while True:
            frame = stream.get_frame()
            if frame is None:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids = detect_markers(gray, dict_name)

            vis = frame.copy()
            if ids is not None and len(ids) > 0:
                cv2.aruco.drawDetectedMarkers(vis, corners, ids)
                poses = estimate_marker_poses(corners, marker_length, camera_matrix, dist_coeffs)

                base_T_gripper = None
                if hand_eye is not None and hand_eye["mode"] == "eye_in_hand" and pose_provider:
                    base_T_gripper = pose_provider.get_pose_matrix()

                flat_ids = ids.flatten().tolist()
                for idx, (marker_id, T) in enumerate(zip(flat_ids, poses)):
                    if T is None:
                        continue
                    draw_axis(vis, camera_matrix, dist_coeffs, T, length=marker_length)
                    x, y, z = T[:3, 3]
                    label = f"ID{marker_id} cam:({x:.3f},{y:.3f},{z:.3f})"

                    if hand_eye is not None:
                        base_T_marker = transform_pose(T, hand_eye, base_T_gripper)
                        if base_T_marker is not None:
                            bx, by, bz = base_T_marker[:3, 3]
                            label = f"base:({bx:.3f},{by:.3f},{bz:.3f})"

                    corner_pt = corners[idx].reshape(4, 2)[0]
                    cv2.putText(vis, label, (int(corner_pt[0]), int(corner_pt[1]) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            cv2.imshow("ArUco Detection", vis)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        stream.stop()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="RealSense ArUco 坐标系检测")
    parser.add_argument("--intrinsics", type=str, required=True)
    parser.add_argument("--hand-eye", type=str, default=None,
                         help="可选：手眼标定结果 json, 用于换算到机器人基坐标系")
    parser.add_argument("--marker-length", type=float, default=0.04,
                         help="marker 边长(米), 独立ArUco marker请按实际测量值传入")
    parser.add_argument("--dict", type=str, default="DICT_4X4_50",
                         help="需与实际使用的marker字典后缀一致(_50/_100/_250/_1000)")
    parser.add_argument("--ip", type=str, default="192.168.58.2",
                         help="机械臂控制器 IP (仅 eye_in_hand 模式需要实时读取关节角时使用)")
    parser.add_argument("--robot-model", type=str, default="FR5",
                         help="FairinoRobot 型号名 (仅 eye_in_hand 模式使用)")
    args = parser.parse_args()

    camera_matrix, dist_coeffs = load_camera_intrinsics(Path(args.intrinsics))
    hand_eye = load_hand_eye(Path(args.hand_eye)) if args.hand_eye else None

    pose_provider = None
    if hand_eye is not None and hand_eye["mode"] == "eye_in_hand":
        import sys
        sys.path.append(str(Path(__file__).resolve().parents[1]))  # 加入 scripts/ 目录
        from robot.fairino import Robot  # 官方 fairino Python SDK
        from robot.fairino_robot import FairinoRobot
        from camera_calibration import RobotPoseProvider

        print(f"[connect] 正在连接机械臂控制器 {args.ip} ...")
        rpc_robot = Robot.RPC(args.ip)
        print("[connect] 连接成功")

        def get_joint_deg_fn() -> list:
            err, joint_deg = rpc_robot.GetActualJointPosDegree()
            if err != 0:
                raise RuntimeError(f"读取当前关节位置失败, 错误码: {err}")
            return joint_deg

        kinematics_model = FairinoRobot(model_name=args.robot_model)
        pose_provider = RobotPoseProvider(kinematics_model, get_joint_deg_fn)

    run_stream(camera_matrix, dist_coeffs, args.marker_length, args.dict,
               hand_eye=hand_eye, pose_provider=pose_provider)


if __name__ == "__main__":
    main()