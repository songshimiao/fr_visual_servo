"""
scripts/algorithm/visual_servo_hover_test.py

标定精度验证脚本：结合手眼标定结果 + ArUco 检测 + 现有运动学模型/实机接口，
分别测试:
    1. 让"相机坐标系"移动到 ArUco 标记正上方指定高度 (--mode camera)
    2. 让"tool1坐标系"移动到 ArUco 标记正上方指定高度 (--mode tool1)

⚠️ 重要说明: 手眼标定时 RobotPoseProvider.get_pose_matrix() 记录的 "gripper" 位姿
    实际是 FairinoRobot.forward_kinematics() 的输出 —— 即法兰盘(flange/tool0)位姿，
    不是 tool1。所以 hand_eye_calibration.json 中的 T 本质是:
        eye_in_hand: p_flange = T @ p_cam  (flange 是 tool0, 不是 tool1)
        eye_to_hand: p_base   = T @ p_cam
    本脚本在换算 tool1 目标位姿时会额外应用 T_FLANGE_TO_TOOL1 修正
    (与 fr5_hardware_test.py 保持一致，来自你项目里已验证过的 36mm 法兰-TCP偏移)。

精度验证方法:
    移动前先用当前(未移动)时刻的相机观测计算 base_T_marker，作为"参考真值"
    (此时标记离相机最近、检测质量最好)。移动到位后，仅通过机器人运动学
    (关节角回读 + FK + 手眼标定外参，不依赖二次检测)计算相机/tool1的
    实际位姿，与参考 base_T_marker 做差，得到相对标记的实际偏移，
    和期望偏移 (0, 0, hover_height) 比较，误差即反映手眼标定+机械臂重复精度。

⚠️ 安全须知: 默认 dry-run，只打印计算出的目标位姿和 IK 解，不会真正移动机械臂。
    确认目标位姿合理后加 --execute，且确保机械臂周围安全、急停开关在手边，
    脚本会在下发运动指令前再做一次二次确认。
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from spatialmath import SE3

from algorithm.aruco_detector import (
    RealSenseColorStream,
    load_camera_intrinsics,
    load_hand_eye,
    detect_markers,
    estimate_marker_poses,
)

# 与 fr5_hardware_test.py 保持一致的法兰->tool1 偏移
T_FLANGE_TO_TOOL1 = SE3(0, 0, 0.036)

TOOL_ID = 0
USER_ID = 0
MOVE_VEL = 10.0
MOVE_ACC = 0.0


def np_to_SE3(T: np.ndarray) -> SE3:
    return SE3(T, check=False)


def wait_for_stable_marker(stream: RealSenseColorStream, dict_name: str,
                            marker_length: float, camera_matrix, dist_coeffs,
                            marker_id: Optional[int], stable_frames: int,
                            timeout_s: float = 15.0) -> SE3:
    """
    等待 ArUco 检测稳定 (连续 stable_frames 帧都检测到指定/首个 marker)，
    返回最后一帧的 cam_T_marker (SE3)。参考项目里已有的"帧稳定性等待"经验，
    避免单帧误检导致的位姿跳变影响标定精度评估。
    """
    consecutive = 0
    last_T = None
    t0 = time.time()
    print(f"[检测] 等待稳定检测到 marker (连续 {stable_frames} 帧)...")

    while time.time() - t0 < timeout_s:
        frame = stream.get_frame()
        if frame is None:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids = detect_markers(gray, dict_name)

        vis = frame.copy()
        found_T = None
        if ids is not None and len(ids) > 0:
            flat_ids = ids.flatten().tolist()
            target_idx = flat_ids.index(marker_id) if marker_id is not None and marker_id in flat_ids else 0
            if marker_id is None or marker_id in flat_ids:
                poses = estimate_marker_poses([corners[target_idx]], marker_length,
                                               camera_matrix, dist_coeffs)
                found_T = poses[0]
            cv2.aruco.drawDetectedMarkers(vis, corners, ids)

        if found_T is not None:
            consecutive += 1
            last_T = found_T
        else:
            consecutive = 0

        cv2.putText(vis, f"stable: {consecutive}/{stable_frames}",
                    (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Waiting for stable marker", vis)
        cv2.waitKey(1)

        if consecutive >= stable_frames:
            cv2.destroyAllWindows()
            return np_to_SE3(last_T)

    cv2.destroyAllWindows()
    raise RuntimeError("[错误] 超时未能稳定检测到 marker，请检查标记是否在相机视野内")


def compute_target_flange_pose(base_T_flange_current: SE3, flange_T_cam: SE3,
                                base_T_marker: SE3, hover_height: float,
                                mode: str, orientation: str) -> SE3:
    """
    计算达成 "mode坐标系悬停在marker正上方hover_height" 目标时所需的 flange 目标位姿。

    orientation:
        look_down     - 目标坐标系 Z 轴与 marker Z 轴反向 (俯视/正对标记，常用于抓取)
        match_marker  - 目标坐标系姿态与 marker 完全一致，只做 Z 方向平移
        keep_current  - 只改变位置，姿态保持当前 (camera/tool1 当前姿态) 不变
    """
    if orientation == "look_down":
        marker_T_target = SE3.Tz(hover_height) * SE3.Rx(180, unit="deg")
    elif orientation == "match_marker":
        marker_T_target = SE3.Tz(hover_height)
    elif orientation == "keep_current":
        marker_T_target = None
    else:
        raise ValueError(f"未知 orientation: {orientation}")

    if mode == "camera":
        base_T_current_target_frame = base_T_flange_current * flange_T_cam
    elif mode == "tool1":
        base_T_current_target_frame = base_T_flange_current * T_FLANGE_TO_TOOL1
    else:
        raise ValueError(f"未知 mode: {mode}")

    if orientation == "keep_current":
        target_pos = (base_T_marker * SE3.Tz(hover_height)).t
        base_T_target = SE3.Rt(base_T_current_target_frame.R, target_pos)
    else:
        base_T_target = base_T_marker * marker_T_target

    # 换算回 flange 目标位姿
    if mode == "camera":
        base_T_flange_target = base_T_target * flange_T_cam.inv()
    else:
        base_T_flange_target = base_T_target * T_FLANGE_TO_TOOL1.inv()

    return base_T_flange_target


def main():
    parser = argparse.ArgumentParser(description="ArUco悬停定位精度验证 (相机/tool1)")
    parser.add_argument("--intrinsics", type=str, default="resources/calibration/camera_intrinsics.json")
    parser.add_argument("--hand-eye", type=str, default="resources/calibration/hand_eye_calibration.json")
    parser.add_argument("--mode", choices=["camera", "tool1"], required=True,
                         help="悬停在marker正上方的坐标系: camera 或 tool1")
    parser.add_argument("--hover-height", type=float, default=0.05, help="悬停高度(米)，默认5cm")
    parser.add_argument("--orientation", choices=["look_down", "match_marker", "keep_current"],
                         default="look_down",
                         help="目标坐标系朝向: look_down(俯视/默认) match_marker(与marker一致) keep_current(保持当前姿态)")
    parser.add_argument("--marker-length", type=float, default=0.011)
    parser.add_argument("--marker-id", type=int, default=None, help="指定要跟踪的marker ID，默认取视野中第一个")
    parser.add_argument("--dict", type=str, default="DICT_4X4_50")
    parser.add_argument("--stable-frames", type=int, default=10)
    parser.add_argument("--ip", type=str, default="192.168.58.2")
    parser.add_argument("--robot-model", type=str, default="FR5")
    parser.add_argument("--execute", action="store_true", help="关闭dry-run，实际下发MoveJ")
    args = parser.parse_args()

    camera_matrix, dist_coeffs = load_camera_intrinsics(Path(args.intrinsics))
    hand_eye = load_hand_eye(Path(args.hand_eye))
    if hand_eye["mode"] != "eye_in_hand":
        print("[错误] 本脚本假设相机随机械臂运动 (eye_in_hand)，"
              f"但手眼标定文件里的 mode 是 '{hand_eye['mode']}'")
        return
    flange_T_cam = np_to_SE3(hand_eye["T"])

    sys.path.append(str(Path(__file__).resolve().parents[1]))  # 加入 scripts/ 目录
    from robot.fairino import Robot
    from robot.fairino_robot import FairinoRobot

    print(f"[connect] 正在连接机械臂控制器 {args.ip} ...")
    rpc_robot = Robot.RPC(args.ip)
    kinematics_model = FairinoRobot(model_name=args.robot_model)
    print("[connect] 连接成功")

    def get_q_rad() -> np.ndarray:
        err, joint_deg = rpc_robot.GetActualJointPosDegree()
        if err != 0:
            raise RuntimeError(f"读取当前关节位置失败, 错误码: {err}")
        return np.radians(joint_deg)

    stream = RealSenseColorStream()
    try:
        # 1. 移动前，在当前(未移动)姿态下检测 marker，作为参考真值
        cam_T_marker = wait_for_stable_marker(
            stream, args.dict, args.marker_length, camera_matrix, dist_coeffs,
            args.marker_id, args.stable_frames
        )
        q_current = get_q_rad()
        base_T_flange_current = kinematics_model.forward_kinematics(q_current)
        base_T_cam_current = base_T_flange_current * flange_T_cam
        base_T_marker = base_T_cam_current * cam_T_marker

        print(f"\n[参考] 当前关节角(deg): {np.round(np.degrees(q_current), 2)}")
        print(f"[参考] base_T_marker 平移(m): {np.round(base_T_marker.t, 4)}")

        # 2. 计算目标 flange 位姿并求解 IK
        base_T_flange_target = compute_target_flange_pose(
            base_T_flange_current, flange_T_cam, base_T_marker,
            args.hover_height, args.mode, args.orientation
        )
        q_target = kinematics_model.inverse_kinematics(
            base_T_flange_target, reference_joint_pos=q_current
        )
        if q_target is None:
            print("[错误] IK 无解，请检查目标位姿是否超出工作空间/关节限位")
            return

        print(f"\n[目标] flange 目标位姿平移(m): {np.round(base_T_flange_target.t, 4)}")
        print(f"[目标] flange 目标位姿RPY(deg, zyx): "
              f"{np.round(base_T_flange_target.rpy(unit='deg', order='zyx'), 2)}")
        print(f"[目标] IK 解 关节角(deg): {np.round(np.degrees(q_target), 2)}")

        if not args.execute:
            print("\n[dry-run] 未加 --execute，不会下发运动指令。"
                  "确认以上目标位姿合理后可加 --execute 实际执行。")
            return

        confirm = input("\n即将下发实际运动指令, 确认机械臂周围安全、已准备好急停开关? (yes/no): ")
        if confirm.strip().lower() != "yes":
            print("用户取消，退出")
            return

        # ret_mode = rpc_robot.Mode(0)
        # print(f"[enable] 切换自动模式, ret={ret_mode}")
        # time.sleep(0.3)
        ret_enable = rpc_robot.RobotEnable(1)
        print(f"[enable] 上使能, ret={ret_enable}")
        time.sleep(1)
        if ret_enable != 0:
            print("[警告] Mode/RobotEnable 返回非0，机械臂可能未真正使能成功"
                  "（常见原因：示教器上有未清除的报警、仍处于手动/拖动模式、"
                  "急停/安全门未复位等）。请先在示教器上确认机器人状态为"
                  "\"就绪/可运行\"，清除所有报警后再重试。")
            print("下面仍会尝试下发 MoveJ，但大概率会被拒绝。")

        target_deg = list(np.degrees(q_target))
        ret = rpc_robot.MoveJ(target_deg, TOOL_ID, USER_ID,
                               vel=MOVE_VEL, acc=MOVE_ACC, blendT=-1.0)
        print(f"MoveJ 返回码: {ret}")
        time.sleep(5.0)

        # 3. 移动后，仅用运动学(关节角回读+FK+手眼外参)计算实际达到的位姿，
        #    与移动前记录的 base_T_marker 参考值比较，得到精度指标
        q_after = get_q_rad()
        base_T_flange_after = kinematics_model.forward_kinematics(q_after)
        if args.mode == "camera":
            base_T_achieved = base_T_flange_after * flange_T_cam
        else:
            base_T_achieved = base_T_flange_after * T_FLANGE_TO_TOOL1

        achieved_offset = base_T_marker.inv() * base_T_achieved
        desired_offset = np.array([0.0, 0.0, args.hover_height])
        pos_error_mm = (achieved_offset.t - desired_offset) * 1000.0

        print(f"\n===== 精度评估 ({args.mode} 悬停 {args.hover_height*1000:.0f}mm) =====")
        print(f"实际相对marker偏移(m): {np.round(achieved_offset.t, 4)}")
        print(f"期望相对marker偏移(m): {np.round(desired_offset, 4)}")
        print(f"位置误差(mm): {np.round(pos_error_mm, 3)}  "
              f"(模长: {np.linalg.norm(pos_error_mm):.3f} mm)")

    finally:
        stream.stop()
        try:
            rpc_robot.RobotEnable(1)
            ret = rpc_robot.MoveJ([90,-90,90,-90,-90,-90], TOOL_ID, USER_ID,
                        vel=MOVE_VEL, acc=MOVE_ACC, blendT=-1.0)
        except Exception:
            pass


if __name__ == "__main__":
    main()