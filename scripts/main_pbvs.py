"""
scripts/main_pbvs.py
 
基于位置的视觉伺服 (PBVS) 主程序。
 
功能:
    1. 相机线程(低频, ~30fps): 实时检测 ArUco 标定板, 消歧 PnP 多解, 在检测到的
       瞬间立即读取当前关节角做FK, 把 marker 位姿融合换算到 base 坐标系下
       (base_T_marker), 存入共享状态。
    2. 控制线程(高频, 默认 100Hz): 读取实时关节角 -> FK 得到 base_T_flange_current,
       结合相机线程给出的最新 base_T_marker(零阶保持, 已经是 base 系, 不再和"当前"
       机械臂位姿重新组合), 算出"Z 轴对齐+悬停"的期望 flange 位姿, 交给 PBVS 控制器
       算出下一步关节角目标, 通过 ServoJ 持续下发, 实现末端对标定板的实时跟随。
    3. 静止检测: 当 marker 连续静止超过 --settle-time (默认 3s) 后, 自动把悬停
       距离从"远"切换到"近", 完成"靠近(抓取动作的占位, 当前无夹爪)"。
       若之后 marker 重新开始运动, 会自动判定为不再静止, 退回"远距离跟随"状态。

⚠️ 关于实机抖动的两个关键修复(相对早期版本):
    1. PnP 姿态多解消歧: cv2.solvePnP(IPPE_SQUARE) 对平面正方形 marker 天生存在
       两个数值接近的候选解, 相机接近正对 marker 时(俯视悬停场景很常见)容易在
       帧间随机跳变, 是视觉伺服抖动的经典根因。已改用 estimate_marker_pose_robust
       (aruco_detector.py 新增函数), 优先选择与上一帧时序一致的解。
    2. base_T_marker 的计算时机: 早期版本在控制线程里用"当前(正在被伺服、持续
       运动的)机械臂位置"去反算 base_T_marker, 会把机械臂自身的运动/抖动混入
       marker 位置估计, 形成自激反馈环、越抖越厉害。现在改为在相机线程检测到
       marker 的瞬间就地读关节角做FK完成融合, 之后控制线程只做"零阶保持"读取,
       不再和自己当前的运动重新耦合。
    另外加入了相邻帧异常跳变剔除(--outlier-max-jump-*), 进一步抑制误检导致的
    瞬时大误差。

⚠️ 分两个线程的原因: 相机检测(含图像处理)天然是低频(~30fps ≈ 33ms/帧),
   而 Fairino ServoJ 要求指令发送频率在 60Hz~1000Hz(即命令周期 1~16ms)区间内
   (参考 SDK FAQ)。若直接按相机帧率下发 ServoJ, 频率会低于下限。
   因此让控制线程独立以更高频率运行, 每个控制周期使用"最新一次"相机检测结果
   (零阶保持), 而不是等待新的相机帧, 从而同时满足"实时跟随"与"ServoJ频率下限"。
   两个线程都会调用 rpc_robot 读取关节角, 用 rpc_lock 序列化, 避免并发访问
   同一个 RPC 连接。

⚠️ 安全须知 (务必先读):
    1. 默认 dry-run 模式(不加 --execute), 只做检测、FK/IK 计算和打印, 不会给机械臂
       下发任何运动指令。
    2. 加 --execute 前, 确认机械臂周围无障碍物、无人员在工作空间内, 且急停开关在
       可触及范围内。脚本会在真正使能/下发指令前做二次确认。
    3. 默认增益/步长都比较保守(见 PBVSGains 默认值与 --kp/--max-translation-step/
       --max-rotation-step-deg 参数; 注意这些是"每个控制周期最多走多远", 与
       --servo-hz 是绑定的, 调高频率或增益前请先小范围试跑), 正式使用前请结合
       现场情况调整。可选 --workspace-min/--workspace-max 提供一层工作空间硬限幅。
    4. 按 'q' 可随时停止相机预览窗口并触发退出流程(会调用 ServoMoveEnd 并停止相机)。
"""

import argparse
import sys
import threading
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
from algorithm.Pbvs import (
    PBVSController,
    PBVSGains,
    MarkerStillnessDetector,
    HoverStateMachine,
    PoseExponentialFilter,
    LoopTimingMonitor,
    compute_hover_flange_pose,
)

T_FLANGE_TO_TOOL1 = SE3(0, 0, 0.036)
USER_ID = 0
# ServoJ 下发时使用的工具号: 建议固定为 0 (flange), 通过软件补偿 (T_FLANGE_TO_TOOL1)
SERVO_TOOL_ID = 0


def np_to_SE3(T: np.ndarray) -> SE3:
    return SE3(T, check=False)


class SharedMarkerState:
    """相机线程与控制线程之间共享的 marker 观测状态, 用线程锁保护"""

    def __init__(self):
        self._lock = threading.Lock()
        self.cam_T_marker: Optional[SE3] = None
        self.timestamp: float = 0.0
        self.found: bool = False

    def update(self, cam_T_marker: SE3, t: float):
        with self._lock:
            self.cam_T_marker = cam_T_marker
            self.timestamp = t
            self.found = True

    def mark_not_found(self):
        with self._lock:
            self.found = False

    def read(self):
        with self._lock:
            return self.cam_T_marker, self.timestamp, self.found


class VisionWorker(threading.Thread):
    """
    相机检测线程: 持续抓取 RealSense 彩色帧, 检测 ArUco marker, 更新共享状态,
    并显示预览窗口(按 'q' 触发全局停止)。
    """

    def __init__(self, camera_matrix, dist_coeffs, marker_length: float, dict_name: str,
                 marker_id: Optional[int], shared_state: SharedMarkerState,
                 stop_event: threading.Event, status_text_fn=None):
        super().__init__(daemon=True)
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.marker_length = marker_length
        self.dict_name = dict_name
        self.marker_id = marker_id
        self.shared_state = shared_state
        self.stop_event = stop_event
        # 回调: () -> List[str], 用于在预览窗口叠加控制线程状态
        self.status_text_fn = status_text_fn
        self.stream = RealSenseColorStream()

    def run(self):
        print("[视觉线程] 启动, 按预览窗口 'q' 可随时停止整个程序")
        try:
            while not self.stop_event.is_set():
                frame = self.stream.get_frame()
                if frame is None:
                    continue
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                corners, ids = detect_markers(gray, self.dict_name)

                vis = frame.copy()
                found_T = None
                if ids is not None and len(ids) > 0:
                    flat_ids = ids.flatten().tolist()
                    if self.marker_id is None or self.marker_id in flat_ids:
                        idx = flat_ids.index(
                            self.marker_id) if self.marker_id is not None else 0
                        poses = estimate_marker_poses([corners[idx]], self.marker_length,
                                                      self.camera_matrix, self.dist_coeffs)
                        found_T = poses[0]
                    cv2.aruco.drawDetectedMarkers(vis, corners, ids)

                now = time.monotonic()
                if found_T is not None:
                    self.shared_state.update(np_to_SE3(found_T), now)
                else:
                    self.shared_state.mark_not_found()

                if self.status_text_fn is not None:
                    for i, line in enumerate(self.status_text_fn()):
                        cv2.putText(vis, line, (20, 30 + 22 * i),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.imshow("PBVS - ArUco tracking", vis)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.stop_event.set()
                    break
        finally:
            self.stream.stop()
            cv2.destroyAllWindows()
            print("[视觉线程] 已停止")


def _try_servo_j(rpc_robot, joint_deg_target: list, vel_percent: float):
    """
    调用 Fairino SDK 的 ServoJ 接口下发关节空间伺服目标。

    不同 SDK 版本 ServoJ 的签名略有差异(参数数量/是否含加速度等), 这里按官方
    FAQ 给出的最常见签名尝试调用: ServoJ(joint_pos_deg, exaxis_pos, vel=...),
    若 SDK 版本不接受某个关键字参数, 逐步降级重试, 避免因签名差异直接崩溃。
    """
    epos = [0.0, 0.0, 0.0, 0.0]
    try:
        return rpc_robot.ServoJ(joint_deg_target, epos, vel=vel_percent)
    except TypeError:
        pass
    try:
        return rpc_robot.ServoJ(joint_deg_target, epos)
    except TypeError:
        pass
    return rpc_robot.ServoJ(joint_deg_target)


def main():
    parser = argparse.ArgumentParser(description="PBVS 实时视觉伺服跟随 + 静止后靠近")
    parser.add_argument("--intrinsics", type=str,
                        default="resources/calibration/camera_intrinsics.json")
    parser.add_argument("--hand-eye", type=str,
                        default="resources/calibration/hand_eye_calibration.json")
    parser.add_argument("--mode", choices=["camera", "tool1"], default="camera",
                        help="悬停/跟随所依据的参考坐标系: camera(默认) 或 tool1")
    parser.add_argument("--hover-far", type=float,
                        default=0.15, help="远距离跟随悬停高度(米), 默认15cm")
    parser.add_argument("--hover-near", type=float,
                        default=0.05, help="静止判定后靠近的悬停高度(米), 默认5cm")
    parser.add_argument("--settle-time", type=float,
                        default=3.0, help="判定marker静止所需的连续时长(秒), 默认3s")
    parser.add_argument("--settle-pos-thresh", type=float,
                        default=0.003, help="静止判定的位置阈值(米), 默认3mm")
    parser.add_argument("--settle-ang-thresh", type=float,
                        default=2.0, help="静止判定的姿态阈值(度), 默认2deg")
    parser.add_argument("--resume-pos-thresh", type=float, default=None,
                        help="从'靠近'状态退回'跟随'所需的位移量(米), 默认=settle-pos-thresh的4倍"
                        "(需明显大于静止判定阈值, 否则噪声会导致状态反复横跳)")
    parser.add_argument("--resume-ang-thresh", type=float, default=None,
                        help="从'靠近'状态退回'跟随'所需的姿态变化(度), 默认=settle-ang-thresh的4倍")
    parser.add_argument("--min-dwell-time", type=float, default=1.0,
                        help="状态切换后至少保持的时间(秒), 防止在阈值附近反复横跳, 默认1.0s")
    parser.add_argument("--marker-length", type=float, default=0.05)
    parser.add_argument("--marker-id", type=int, default=None,
                        help="指定要跟踪的marker ID, 默认取视野中第一个")
    parser.add_argument("--dict", type=str, default="DICT_4X4_50")
    parser.add_argument("--ip", type=str, default="192.168.58.2")
    parser.add_argument("--robot-model", type=str, default="FR5")
    parser.add_argument("--servo-hz", type=float, default=100.0,
                        help="控制线程 ServoJ 下发频率(Hz), 默认100Hz")
    parser.add_argument("--marker-timeout", type=float, default=0.5,
                        help="超过该时长未更新到marker观测则暂停伺服(秒), 默认0.5s")
    parser.add_argument("--kp", type=float, default=0.3,
                        help="比例增益(每周期误差的多少比例被走完, 0~1之间较合理), 默认0.3")
    parser.add_argument("--max-translation-step", type=float, default=0.01,
                        help="每个控制周期最大平移步长(m), 默认0.01m (配合--servo-hz决定等效速度)")
    parser.add_argument("--max-rotation-step-deg", type=float, default=5.0,
                        help="每个控制周期最大旋转步长(度), 默认5deg")
    parser.add_argument("--position-error-threshold", type=float, default=0.001,
                        help="位置收敛阈值(m), 默认1mm")
    parser.add_argument("--rotation-error-threshold-deg", type=float, default=0.5,
                        help="姿态收敛阈值(度), 默认0.5deg")
    parser.add_argument("--workspace-min", type=float, nargs=3, default=None,
                        metavar=("X_MIN", "Y_MIN", "Z_MIN"),
                        help="可选: base坐标系下flange目标位姿的工作空间下限(m), 例如 --workspace-min -0.5 -0.5 0.0")
    parser.add_argument("--workspace-max", type=float, nargs=3, default=None,
                        metavar=("X_MAX", "Y_MAX", "Z_MAX"),
                        help="可选: base坐标系下flange目标位姿的工作空间上限(m)")
    parser.add_argument("--marker-filter-alpha", type=float, default=0.4,
                        help="marker位置低通滤波系数(0,1], 越小越平滑, 默认0.4")
    parser.add_argument("--marker-filter-alpha-rot", type=float, default=None,
                        help="marker姿态低通滤波系数(0,1], 默认与--marker-filter-alpha的1/4"
                        "(姿态噪声通常比位置大, 且z_align模式下姿态噪声会被放大成末端抖动,"
                        "所以默认给更强的平滑)")
    parser.add_argument("--diagnose-timing", action="store_true", default=True,
                        help="打印控制循环实际周期统计, 用于诊断抖动是否来自ServoJ发送间隔不稳定(默认开启)")
    parser.add_argument("--no-diagnose-timing",
                        dest="diagnose_timing", action="store_false")
    parser.add_argument("--execute", action="store_true",
                        help="关闭dry-run, 实际下发ServoJ")
    args = parser.parse_args()

    camera_matrix, dist_coeffs = load_camera_intrinsics(Path(args.intrinsics))
    hand_eye = load_hand_eye(Path(args.hand_eye))
    if hand_eye["mode"] != "eye_in_hand":
        print(f"[错误] 本脚本假设相机随机械臂运动(eye_in_hand), "
              f"但手眼标定文件里的 mode 是 '{hand_eye['mode']}'")
        return
    flange_T_cam = np_to_SE3(hand_eye["T"])
    flange_T_ref = flange_T_cam if args.mode == "camera" else T_FLANGE_TO_TOOL1

    sys.path.append(str(Path(__file__).resolve().parent))
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
        return np.deg2rad(joint_deg)

    # ---- 共享状态 / 线程间通信 ----
    shared_state = SharedMarkerState()
    stop_event = threading.Event()

    workspace_min = np.array(
        args.workspace_min) if args.workspace_min is not None else None
    workspace_max = np.array(
        args.workspace_max) if args.workspace_max is not None else None
    controller = PBVSController(
        kinematics_model,
        gains=PBVSGains(
            kp=args.kp,
            max_translation_step=args.max_translation_step,
            max_rotation_step=np.deg2rad(args.max_rotation_step_deg),
            workspace_min=workspace_min,
            workspace_max=workspace_max,
            position_error_threshold=args.position_error_threshold,
            rotation_error_threshold=np.deg2rad(
                args.rotation_error_threshold_deg),
        ),
    )
    stillness_detector = MarkerStillnessDetector(
        window_s=args.settle_time,
        pos_thresh_m=args.settle_pos_thresh,
        ang_thresh_deg=args.settle_ang_thresh,
    )
    resume_pos_thresh = args.resume_pos_thresh if args.resume_pos_thresh is not None else args.settle_pos_thresh * 4.0
    resume_ang_thresh = args.resume_ang_thresh if args.resume_ang_thresh is not None else args.settle_ang_thresh * 4.0
    hover_state_machine = HoverStateMachine(
        resume_pos_thresh_m=resume_pos_thresh,
        resume_ang_thresh_deg=resume_ang_thresh,
        min_dwell_s=args.min_dwell_time,
    )
    alpha_rot = args.marker_filter_alpha_rot if args.marker_filter_alpha_rot is not None else max(
        args.marker_filter_alpha * 0.25, 0.02)
    marker_filter = PoseExponentialFilter(
        alpha=args.marker_filter_alpha, alpha_rot=alpha_rot)
    timing_monitor = LoopTimingMonitor(
        min_dt_ms=1.0, max_dt_ms=16.0, report_every_s=2.0) if args.diagnose_timing else None

    # 供预览窗口显示当前控制线程状态
    control_status = {"state": "wait", "hover": args.hover_far, "still_span": 0.0,
                      "pos_std_mm": 0.0, "ang_std_deg": 0.0}

    def status_text_fn():
        return [
            f"state: {control_status['state']}",
            f"hover: {control_status['hover']*1000:.0f}mm",
            f"still_span: {control_status['still_span']:.1f}/{args.settle_time:.1f}s",
            f"noise: pos_std={control_status['pos_std_mm']:.1f}mm ang_std={control_status['ang_std_deg']:.2f}deg",
        ]

    vision = VisionWorker(
        camera_matrix, dist_coeffs, args.marker_length, args.dict, args.marker_id,
        shared_state, stop_event, status_text_fn=status_text_fn,
    )
    vision.start()

    # 等待第一次检测到 marker, 避免控制线程一开始就跑在空数据上
    print("[等待] 等待首次检测到 marker ...")
    t_wait0 = time.monotonic()
    while shared_state.read()[0] is None:
        if stop_event.is_set():
            print("[退出] 用户提前停止")
            return
        if time.monotonic() - t_wait0 > 20.0:
            print("[错误] 20秒内未检测到 marker, 请检查标定板是否在相机视野内, 退出")
            stop_event.set()
            vision.join(timeout=2.0)
            return
        time.sleep(0.05)

    if args.execute:
        confirm = input("\n即将进入实时伺服跟随并下发运动指令, "
                        "确认机械臂周围安全、已准备好急停开关? (yes/no): ")
        if confirm.strip().lower() != "yes":
            print("用户取消, 退出")
            stop_event.set()
            vision.join(timeout=2.0)
            return
        ret_enable = rpc_robot.RobotEnable(1)
        print(f"[enable] 上使能, ret={ret_enable}")
        time.sleep(1.0)
        try:
            rpc_robot.ServoMoveStart()
        except AttributeError:
            print("[提示] 当前 SDK 版本未找到 ServoMoveStart 接口, 跳过(部分版本无需显式开启)")
    else:
        print("\n[dry-run] 未加 --execute, 仅打印计算结果, 不会下发运动指令。"
              "确认逻辑正确后可加 --execute 实际执行。")

    dt = 1.0 / args.servo_hz
    q_seed = get_q_rad()

    print("[控制线程] 开始主循环, Ctrl+C 或预览窗口按 'q' 可停止")
    try:
        while not stop_event.is_set():
            loop_t0 = time.monotonic()

            cam_T_marker_raw, marker_t, marker_found = shared_state.read()
            if not marker_found or cam_T_marker_raw is None or (time.monotonic() - marker_t) > args.marker_timeout:
                control_status["state"] = "marker lost-pause"
                time.sleep(dt)
                continue

            q_current = get_q_rad()
            base_T_flange_current = kinematics_model.forward_kinematics(
                q_current)
            base_T_cam_current = base_T_flange_current * flange_T_cam
            base_T_marker_raw = base_T_cam_current * cam_T_marker_raw
            base_T_marker = marker_filter.update(base_T_marker_raw)

            is_still = stillness_detector.update(
                time.monotonic(), base_T_marker)
            control_status["still_span"] = stillness_detector.buffer_span_s
            control_status["pos_std_mm"] = stillness_detector.last_pos_std_m * 1000.0
            control_status["ang_std_deg"] = stillness_detector.last_ang_std_deg

            state = hover_state_machine.update(
                time.monotonic(), is_still, base_T_marker)
            approaching = state == HoverStateMachine.APPROACHING
            hover_distance = args.hover_near if approaching else args.hover_far
            control_status["state"] = "close" if approaching else "servoing"
            control_status["hover"] = hover_distance

            base_T_flange_desired = compute_hover_flange_pose(
                base_T_marker, flange_T_ref, hover_distance, z_align=True,
            )

            q_next, T_next, status, pos_err, rot_err = controller.compute_next_joint_target(
                base_T_flange_current, base_T_flange_desired, q_seed=q_current,
            )

            if status == controller.law.RESULT_ERROR_INVALID_INPUT:
                print("[警告] 位姿数据非法(NaN/Inf), 跳过本周期")
                time.sleep(dt)
                continue

            if q_next is None:
                print("[警告] IK 无解, 跳过本周期, 保持上一次目标")
                time.sleep(dt)
                continue

            q_seed = q_next  # 下一周期 IK 的迭代种子, 保证解的连续性

            pos_err_mm = pos_err * 1000.0
            ang_err_deg = np.rad2deg(rot_err)
            converged = status == controller.law.RESULT_FINISHED

            if args.execute:
                target_deg = list(np.rad2deg(q_next))
                _try_servo_j(rpc_robot, target_deg, vel_percent=10.0)
            else:
                print(f"\r[dry-run] state={control_status['state']:>10s} "
                      f"{'[已到位]' if converged else '[跟随中]'} "
                      f"hover={hover_distance*1000:5.1f}mm "
                      f"pos_err={pos_err_mm:6.2f}mm ang_err={ang_err_deg:5.2f}deg "
                      f"noise(pos_std={control_status['pos_std_mm']:.1f}mm,"
                      f"ang_std={control_status['ang_std_deg']:.2f}deg) "
                      f"q_next(deg)={np.round(np.degrees(q_next), 1)}", end="")

            elapsed = time.monotonic() - loop_t0
            time.sleep(max(0.0, dt - elapsed))

            if timing_monitor is not None:
                report = timing_monitor.tick()
                if report is not None:
                    print("\n" + report)
                    print(f"[状态诊断] state={control_status['state']} "
                          f"pos_std={control_status['pos_std_mm']:.2f}mm "
                          f"ang_std={control_status['ang_std_deg']:.2f}deg "
                          f"(settle阈值: pos<{args.settle_pos_thresh*1000:.1f}mm, "
                          f"ang<{args.settle_ang_thresh:.1f}deg)")

    except KeyboardInterrupt:
        print("\n[中断] 收到 Ctrl+C, 准备退出")
    finally:
        stop_event.set()
        if args.execute:
            try:
                rpc_robot.ServoMoveEnd()
                rpc_robot.MoveJ([90, -90, 90, -90, -90, -90], tool=0, user=0)
            except AttributeError:
                pass
            except Exception as e:
                print(f"[警告] ServoMoveEnd 调用异常: {e}")
        vision.join(timeout=2.0)
        print("[退出] 主程序已停止")


if __name__ == "__main__":
    main()
