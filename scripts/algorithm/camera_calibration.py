"""
scripts/algorithm/camera_calibration.py

Gemini 305 eye-in-hand / Gemini 335 eye-to-hand 双相机标定模块（保留 RealSense）
1. 内参标定: 使用 ChArUco 标定板 (cv2.aruco.calibrateCameraCharuco)
2. 手眼标定: 支持 Eye-in-Hand (AX=XB) 与 Eye-to-Hand (AX=ZB), 标定结果保存为 json

依赖:
    pip install opencv-contrib-python pyrealsense2 numpy

用法:
    1. 采集 + 内参标定
    python camera_calibration.py intrinsic --num-images 20 \
        --save-images-dir resources/intrinsics_img/
        --out resources/calibration/camera_intrinsics.json

    2. 手眼标定 (需要机械臂在线, 且已完成内参标定), mode 手动选择
    python camera_calibration.py extrinsic --mode eye_in_hand --num-poses 15 \
        --intrinsics resources/calibration/camera_intrinsics.json \
        --out resources/calibration/hand_eye_calibration.json

    python camera_calibration.py extrinsic --mode eye_to_hand --num-poses 15 \
        --intrinsics resources/camera_intrinsics.json \
        --out resources/hand_eye_calibration.json
"""

import argparse
import json
import time
import threading
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np

if __package__:
    from .camera_stream import (RealSenseColorStream, create_camera, add_camera_arguments,
                                resolve_camera_arguments, validate_calibration, paired_camera_settings,
                                calibration_path, camera_role)
    from .web_preview import WebPreview, add_web_arguments
else:
    from camera_stream import (RealSenseColorStream, create_camera, add_camera_arguments,
                               resolve_camera_arguments, validate_calibration, paired_camera_settings,
                                calibration_path, camera_role)
    from web_preview import WebPreview, add_web_arguments


# --------------------------------------------------------------------------- #
# ChArUco 标定板配置 —— 按实际标定板参数设置 (5x7, 15mm/11mm, DICT_4X4)
# 注意: DICT_4X4 家族在 OpenCV 里必须带具体数量后缀 (_50/_100/_250/_1000)，
#       这取决于生成这块板子时选用的字典大小。5x7 板最多用到 17 个 marker，
#       DICT_4X4_50 足够覆盖；如果生成时用的是其他后缀（如 _100/_1000），
#       请务必改成一致的，否则检测/标定会失败或错位。
# --------------------------------------------------------------------------- #
CHARUCO_SQUARES_X = 7
CHARUCO_SQUARES_Y = 5
CHARUCO_SQUARE_LENGTH = 0.015   # 单位: 米，棋盘格边长 (15mm)
CHARUCO_MARKER_LENGTH = 0.011   # 单位: 米，ArUco 标记边长 (11mm)
ARUCO_DICT_NAME = "DICT_4X4_50"


def _get_aruco_dict(name: str):
    dict_id = getattr(cv2.aruco, name)
    return cv2.aruco.getPredefinedDictionary(dict_id)


def _make_charuco_board():
    """创建 ChArUco Board"""
    aruco_dict = _get_aruco_dict(ARUCO_DICT_NAME)
    size = (CHARUCO_SQUARES_X, CHARUCO_SQUARES_Y)
    if hasattr(cv2.aruco, "CharucoBoard"):
        # OpenCV >= 4.7
        board = cv2.aruco.CharucoBoard(
            size, CHARUCO_SQUARE_LENGTH, CHARUCO_MARKER_LENGTH, aruco_dict
        )
    else:
        print(f"[OpenCV] 版本过低, OpenCV >= 4.7")

    return board, aruco_dict


def _detect_charuco(gray, board, aruco_dict):
    """检测 ChArUco 角点, 返回 (charuco_corners, charuco_ids)"""
    if hasattr(cv2.aruco, "ArucoDetector"):
        params = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(aruco_dict, params)
        corners, ids, _ = detector.detectMarkers(gray)
    else:
        print(f"[Opencv] 版本过低, OpenCV >=4.7")

    if ids is None or len(ids) == 0:
        return None, None

    ok, ch_corners, ch_ids = cv2.aruco.interpolateCornersCharuco(
        corners, ids, gray, board
    )
    if not ok or ch_corners is None or len(ch_corners) < 4:
        return None, None
    return ch_corners, ch_ids


# Backward-compatible public name; CLI creation uses the selected backend.
RealSenseCamera = RealSenseColorStream


# -------------------------
# 1. 内参标定
# -------------------------
def collect_charuco_images(camera: RealSenseCamera, board, aruco_dict,
                           num_images: int, save_dir: Optional[Path] = None, *, preview: WebPreview
                           ) -> Tuple[list, list, Tuple[int, int]]:
    """
    实时采集 ChArUco 图像。按 's' 保存当前帧用于标定，按 'q' 提前结束。
    返回: all_charuco_corners, all_charuco_ids, image_size(w,h)
    """
    all_corners, all_ids = [], []
    image_size = None
    saved = 0

    print(f"[采集] 目标 {num_images} 张有效图像. 在网页中采集/结束 (s/q).")
    preview.set_feedback(saved=0, target=num_images, message="等待保存 ChArUco 样本")
    while saved < num_images and not preview.stop_event.is_set():
        frame = camera.get_color_frame()
        if frame is None:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        image_size = (gray.shape[1], gray.shape[0])

        ch_corners, ch_ids = _detect_charuco(gray, board, aruco_dict)
        vis = frame.copy()
        if ch_corners is not None:
            cv2.aruco.drawDetectedCornersCharuco(vis, ch_corners, ch_ids)

        cv2.putText(vis, f"Saved: {saved}/{num_images}  [s]=save [q]=quit",
                    (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        preview.publish(vis)
        key = preview.poll_key()

        if key == ord('s') and ch_corners is not None:
            if save_dir:
                save_dir.mkdir(parents=True, exist_ok=True)
                image_path = save_dir / f"charuco_{saved + 1:03d}.png"
                if not cv2.imwrite(str(image_path), frame):
                    raise RuntimeError(f"无法保存标定图像: {image_path}")
            all_corners.append(ch_corners)
            all_ids.append(ch_ids)
            saved += 1
            preview.set_feedback(saved=saved, message="样本已保存")
            print(f"  已采集 {saved}/{num_images}")
        elif key == ord('s'):
            preview.set_feedback(message="未保存：未检测到足够的 ChArUco 角点")
        elif key == ord('q'):
            break

    return all_corners, all_ids, image_size


def calibrate_intrinsics(all_corners, all_ids, board, image_size) -> dict:
    """执行 ChArUco 内参标定, 返回内参结果字典"""
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
        charucoCorners=all_corners,
        charucoIds=all_ids,
        board=board,
        imageSize=image_size,
        cameraMatrix=None,
        distCoeffs=None,
    )
    result = {
        "reprojection_error": float(ret),
        "image_size": list(image_size),
        "camera_matrix": camera_matrix.tolist(),
        "dist_coeffs": dist_coeffs.flatten().tolist(),
        "fx": float(camera_matrix[0, 0]),
        "fy": float(camera_matrix[1, 1]),
        "cx": float(camera_matrix[0, 2]),
        "cy": float(camera_matrix[1, 2]),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    print(f"[内参标定完成] 重投影误差 = {ret:.4f} px")
    return result


# ----------------------------------------------------------
# 2. 手眼标定 (305: Eye-in-Hand / 335: Eye-to-Hand)
# ----------------------------------------------------------
class RobotPoseProvider:
    """
    机械臂末端位姿读取适配器。

    不假设具体的硬件 SDK 接口 —— 只需要传入一个"读取当前 6 个关节角(度)"的
    无参回调函数，位姿计算复用项目里已有的 FairinoRobot.forward_kinematics
    (基于 roboticstoolbox, 返回 spatialmath.SE3)，这样标定用的运动学模型
    与现有 IK/FK 保持完全一致，不会因为重新实现正运动学引入误差或约定不一致。
    """

    def __init__(self, kinematics_model, get_joint_deg_fn: Callable[[], List[float]]):
        self.kinematics_model = kinematics_model
        self.get_joint_deg_fn = get_joint_deg_fn

    def get_pose_matrix(self) -> np.ndarray:
        """返回 4x4 齐次变换矩阵: base_T_gripper (末端相对机器人基坐标系)"""
        q_deg = np.asarray(self.get_joint_deg_fn(), dtype=float)
        q_rad = np.deg2rad(q_deg)
        se3_pose = self.kinematics_model.forward_kinematics(q_rad)
        return np.asarray(se3_pose.A)


def estimate_target_pose_in_camera(gray, board, aruco_dict, camera_matrix, dist_coeffs
                                   ) -> Optional[np.ndarray]:
    """检测 ChArUco 角点并估计标定板相对相机的位姿, 返回 4x4 矩阵 cam_T_target"""
    ch_corners, ch_ids = _detect_charuco(gray, board, aruco_dict)
    if ch_corners is None:
        return None

    ok, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
        ch_corners, ch_ids, board, camera_matrix, dist_coeffs, None, None
    )
    if not ok:
        return None

    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = tvec.flatten()
    return T


def collect_hand_eye_samples(camera: RealSenseCamera, board, aruco_dict,
                             camera_matrix, dist_coeffs,
                             pose_provider: RobotPoseProvider,
                             num_poses: int, *, preview: WebPreview
                             ) -> Tuple[list, list, list, list]:
    """
    交互式采集手眼标定样本。
    每次移动机械臂到不同姿态、让标定板在相机视野内保持可见，按 's' 采集一组样本。
    返回: R_gripper2base_list, t_gripper2base_list, R_target2cam_list, t_target2cam_list
    """
    R_g2b, t_g2b, R_t2c, t_t2c = [], [], [], []
    saved = 0
    print(f"[采集] 手眼标定需要 {num_poses} 组不同姿态样本。")
    print("       每次移动机械臂后，在网页中采集一帧(s)，或结束采集(q)。")
    print("       建议：姿态尽量覆盖不同的旋转方向，避免运动轨迹退化。")

    preview.set_feedback(saved=0, target=num_poses, message="等待保存手眼样本；请先使机械臂静止")
    while saved < num_poses and not preview.stop_event.is_set():
        frame = camera.get_color_frame()
        if frame is None:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cam_T_target = estimate_target_pose_in_camera(
            gray, board, aruco_dict, camera_matrix, dist_coeffs
        )

        vis = frame.copy()
        status = "标定板: 已检测" if cam_T_target is not None else "标定板: 未检测"
        cv2.putText(vis, f"{status}  Saved:{saved}/{num_poses}  [s]=save [q]=quit",
                    (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        preview.publish(vis)
        key = preview.poll_key()

        if key == ord('s'):
            if cam_T_target is None:
                preview.set_feedback(message="未保存：未检测到标定板")
                print("  [跳过] 未检测到标定板")
                continue
            base_T_gripper = pose_provider.get_pose_matrix()
            R_g2b.append(base_T_gripper[:3, :3])
            t_g2b.append(base_T_gripper[:3, 3])
            R_t2c.append(cam_T_target[:3, :3])
            t_t2c.append(cam_T_target[:3, 3])
            saved += 1
            preview.set_feedback(saved=saved, message="图像位姿和当前法兰位姿已采集")
            print(f"  已采集 {saved}/{num_poses}")
        elif key == ord('q'):
            break

    return R_g2b, t_g2b, R_t2c, t_t2c


def calibrate_hand_eye(R_g2b, t_g2b, R_t2c, t_t2c, mode: str,
                       method=cv2.CALIB_HAND_EYE_TSAI) -> dict:
    """
    执行手眼标定。

    mode == "eye_in_hand": 相机固定在末端法兰上，输入 gripper2base,
                            求解结果 T 满足: p_flange = T @ p_cam
                            即"将相机坐标系下的点变换到法兰(tool0)坐标系",
                            注意这里的 gripper 特指 FairinoRobot.forward_kinematics()
                            输出的法兰(tool0)坐标系，不是 tool1 (如需 tool1),
                            请另外乘以 T_FLANGE_TO_TOOL1 修正，见 fr5_hardware_test.py）
    mode == "eye_to_hand": 相机固定在外部机架上，将 gripper2base 取逆
                            变为 base2gripper 输入 (AX=ZB 技巧),
                            求解结果 T 满足: p_base = T @ p_cam
                            即"将相机坐标系下的点变换到机器人基坐标系"，可直接使用
    """
    if mode == "eye_in_hand":
        R_in, t_in = R_g2b, t_g2b
    elif mode == "eye_to_hand":
        R_in, t_in = [], []
        for R, t in zip(R_g2b, t_g2b):
            R_inv = R.T
            t_inv = -R_inv @ t
            R_in.append(R_inv)
            t_in.append(t_inv)
    else:
        raise ValueError("mode 必须是 'eye_in_hand' 或 'eye_to_hand'")

    R_cam2x, t_cam2x = cv2.calibrateHandEye(
        R_in, t_in, R_t2c, t_t2c, method=method
    )

    T = np.eye(4)
    T[:3, :3] = R_cam2x
    T[:3, 3] = t_cam2x.flatten()

    result = {
        "mode": mode,
        "num_samples": len(R_g2b),
        "method": "TSAI",
        "cam_T_gripper_or_base": T.tolist(),
        "description": (
            "T满足 p_flange = T @ p_cam：将相机坐标系下的点变换到法兰(tool0)坐标系"
            " (eye_in_hand，注意是 tool0 不是 tool1)"
            if mode == "eye_in_hand" else
            "T满足 p_base = T @ p_cam：将相机坐标系下的点变换到机器人基坐标系"
            " (eye_to_hand，可直接使用)"
        ),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    return result


# --------------
# JSON 读写工具
# --------------
def save_json(data: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"[已保存] {path}")


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_intrinsics_for_cv(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    data = load_json(path)
    camera_matrix = np.array(data["camera_matrix"], dtype=np.float64)
    dist_coeffs = np.array(data["dist_coeffs"], dtype=np.float64)
    return camera_matrix, dist_coeffs


def run_dual_calibration(args, primary_settings, secondary_settings):
    """Each camera owns its reader, samples, intrinsics and physical hand-eye mode.

    The HTTP thread only signals requests. The acquisition worker reads the
    current image/robot pose; cameras are independent and are not synchronized.
    """
    settings_list = [primary_settings, secondary_settings]
    intrinsic_paths = [getattr(args, "intrinsics", None),
                       calibration_path(secondary_settings.model)]
    output_paths = [args.out, calibration_path(secondary_settings.model,
                    "camera_intrinsics" if args.cmd == "intrinsic" else "hand_eye_calibration")]
    if Path(output_paths[0]).resolve() == Path(output_paths[1]).resolve():
        raise ValueError("两台相机的标定输出文件不能相同")
    matrices = []
    pose_provider = None
    if args.cmd == "extrinsic":
        for settings, path in zip(settings_list, intrinsic_paths):
            validate_calibration(path, settings)
            matrices.append(load_intrinsics_for_cv(Path(path)))
        import sys
        sys.path.append(str(Path(__file__).resolve().parents[1]))
        from robot.fairino import Robot
        from robot.fairino_robot import FairinoRobot
        print(f"[connect] 读取机械臂 {args.ip} 的当前关节角（不下发运动）")
        rpc = Robot.RPC(args.ip)
        pose_lock = threading.Lock()

        def get_joint_deg():
            with pose_lock:
                err, joints = rpc.GetActualJointPosDegree()
            if err != 0:
                raise RuntimeError(f"读取当前关节位置失败: {err}")
            return joints

        pose_provider = RobotPoseProvider(FairinoRobot(model_name=args.robot_model), get_joint_deg)
        print("[安装约定] 305: 标定板相对基座固定；335: 标定板相对法兰固定。")
        print("[采集] 两路样本独立保存；请按对应安装约定分别采集，不能混用板的固定方式。")

    streams = []
    errors = []
    workers = []
    completed = [threading.Event() for _ in settings_list]
    with WebPreview.from_args(args, title="双相机标定", allow_capture=True) as preview:
        try:
            for index, settings in enumerate(settings_list):
                paths = [intrinsic_paths[index]] if args.cmd == "extrinsic" else []
                streams.append(create_camera(settings, paths))

            def collect(index, channel):
                camera = streams[index]
                settings = settings_list[index]
                try:
                    board, dictionary = _make_charuco_board()
                    if args.cmd == "intrinsic":
                        root = Path(args.save_images_dir or "resources/intrinsics_img")
                        save_dir = root / settings.model / (time.strftime("%Y%m%d_%H%M%S") + f"_{time.time_ns() % 1000000000:09d}")
                        corners, ids, size = collect_charuco_images(
                            camera, board, dictionary, args.num_images, save_dir, preview=channel)
                        if len(corners) < 4:
                            channel.set_feedback(message="有效样本不足 4 张，未写入标定文件")
                            print(f"[{settings.model}] 有效样本不足，未写入标定文件")
                            return
                        result = calibrate_intrinsics(corners, ids, board, size)
                    else:
                        samples = collect_hand_eye_samples(
                            camera, board, dictionary, *matrices[index], pose_provider,
                            args.num_poses, preview=channel)
                        if len(samples[0]) < 3:
                            channel.set_feedback(message="有效样本不足 3 组，未写入标定文件")
                            print(f"[{settings.model}] 有效样本不足，未写入标定文件")
                            return
                        result = calibrate_hand_eye(*samples, camera_role(settings.model))
                    result["camera"] = camera.metadata
                    save_json(result, Path(output_paths[index]))
                    channel.set_feedback(message=f"标定完成：{output_paths[index]}")
                except Exception as exc:
                    errors.append(exc)
                    channel.set_feedback(message=f"标定失败：{exc}")
                    preview.stop_event.set()
                finally:
                    completed[index].set()
                    # Keep the completed camera live while the other camera is
                    # still collecting its independently mounted board.
                    try:
                        while not preview.stop_event.is_set() and not all(done.is_set() for done in completed):
                            frame = camera.get_color_frame()
                            if frame is not None:
                                channel.publish(frame)
                    except Exception as exc:
                        errors.append(exc)
                        channel.set_feedback(message=f"相机异常：{exc}")
                        preview.stop_event.set()

            for index, channel in enumerate(preview.channels.values()):
                worker = threading.Thread(target=collect, args=(index, channel), daemon=True)
                workers.append(worker)
                worker.start()
            while any(worker.is_alive() for worker in workers):
                for worker in workers:
                    worker.join(timeout=0.1)
            if errors:
                raise RuntimeError(f"双相机标定失败: {errors[0]}") from errors[0]
        finally:
            preview.stop_event.set()
            for worker in workers:
                worker.join()
            for stream in streams:
                stream.stop()


# -------------
# CLI
# -------------
def main():
    parser = argparse.ArgumentParser(description="Gemini 305 eye-in-hand / Gemini 335 eye-to-hand 双相机标定")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("intrinsic", help="内参标定")
    add_web_arguments(p1)
    add_camera_arguments(p1)
    p1.add_argument("--num-images", type=int, default=20)
    p1.add_argument("--out", type=str, default=None)
    p1.add_argument("--save-images-dir", type=str, default=None)

    p2 = sub.add_parser("extrinsic", help="手眼标定 (手动选择 eye_in_hand / eye_to_hand)")
    add_web_arguments(p2)
    add_camera_arguments(p2)
    p2.add_argument("--mode", choices=["eye_in_hand", "eye_to_hand"], default=None)
    p2.add_argument("--num-poses", type=int, default=15)
    p2.add_argument("--intrinsics", type=str, default=None)
    p2.add_argument("--out", type=str, default=None)
    p2.add_argument("--robot-model", type=str, default="FR5", help="FairinoRobot 型号名")
    p2.add_argument("--ip", type=str, default="192.168.58.2", help="机械臂控制器 IP")

    args = parser.parse_args()
    camera_settings = resolve_camera_arguments(args)
    if (args.num_images if args.cmd == "intrinsic" else args.num_poses) < (4 if args.cmd == "intrinsic" else 3):
        parser.error("内参至少 4 张图像；手眼至少 3 组位姿")
    secondary_settings = paired_camera_settings(args)
    if secondary_settings is not None:
        run_dual_calibration(args, camera_settings, secondary_settings)
        return
    if args.cmd == "extrinsic" and args.mode is None:
        if camera_settings.backend == "orbbec":
            args.mode = camera_role(camera_settings.model)
        else:
            parser.error("RealSense 手眼标定需要 --mode")
    if args.cmd == "extrinsic":
        validate_calibration(args.intrinsics, camera_settings)
    board, aruco_dict = _make_charuco_board()

    with WebPreview.from_args(args, title="Camera calibration", allow_capture=True) as preview:
        if args.cmd == "intrinsic":
            camera = create_camera(camera_settings)
            try:
                save_dir = Path(args.save_images_dir) if args.save_images_dir else None
                corners, ids, size = collect_charuco_images(
                    camera, board, aruco_dict, args.num_images, save_dir, preview=preview
                )
                if len(corners) < 4:
                    print("[错误] 有效样本不足, 至少需要 4 张不同角度的图像")
                    return
                result = calibrate_intrinsics(corners, ids, board, size)
                result["camera"] = camera.metadata
                save_json(result, Path(args.out))
            finally:
                camera.stop()

        elif args.cmd == "extrinsic":
            import sys
            sys.path.append(str(Path(__file__).resolve().parents[1]))
            from robot.fairino import Robot
            from robot.fairino_robot import FairinoRobot

            print(f"[connect] 正在连接机械臂控制器 {args.ip} ...")
            rpc_robot = Robot.RPC(args.ip)
            print("[connect] 连接成功")

            def get_joint_deg_fn() -> list:
                err, joint_deg = rpc_robot.GetActualJointPosDegree()
                if err != 0:
                    raise RuntimeError(f"读取当前关节位置失败")
                return joint_deg

            camera_matrix, dist_coeffs = load_intrinsics_for_cv(Path(args.intrinsics))
            camera = create_camera(camera_settings, [args.intrinsics])
            kinematics_model = FairinoRobot(model_name=args.robot_model)
            pose_provider = RobotPoseProvider(kinematics_model, get_joint_deg_fn)

            try:
                R_g2b, t_g2b, R_t2c, t_t2c = collect_hand_eye_samples(
                    camera, board, aruco_dict, camera_matrix, dist_coeffs,
                    pose_provider, args.num_poses, preview=preview
                )
                if len(R_g2b) < 3:
                    print("[错误] 有效样本不足，至少需要 3 组不同姿态")
                    return
                result = calibrate_hand_eye(R_g2b, t_g2b, R_t2c, t_t2c, args.mode)
                result["camera"] = camera.metadata
                save_json(result, Path(args.out))
            finally:
                camera.stop()


if __name__ == "__main__":
    main()
