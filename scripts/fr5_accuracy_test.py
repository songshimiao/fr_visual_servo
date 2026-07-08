"""
FR5 精度 / 重复定位精度测试
===============================

在 `fr5_hardware_test.py` 已经验证 "模型 FK 与实机一致" 的基础上, 此脚本设计实际动作, 量化评估:
1. 绝对精度 (Accuracy):
    在工作空间内选取若干个关节空间目标点, 实际运动过后, 对比:
        - 模型 FK (实际关节角) 计算出的位姿
        - 控制器反馈的实际 TCP 位姿
    得到每个点的 平移误差 (mm) 和 旋转误差 (deg)。

2. 重复定位精度 (Repeatability, 参考 ISO 9283 思路):
    选取 1-N 个目标点, 每个点重复运动 N 次,
    记录每次到位后控制器反馈的 TCP 位姿, 计算位姿相对其均值的离散程度:
        - 位置重复精度 RPl = mean(dist_i) + 3 * std(dist_i)     (dist_i = ||p_i - p_mean||)
        - 姿态重复精度用旋转误差的 std / max 表示
"""

import argparse
import sys
import time
import json

import numpy as np

from spatialmath import SE3
from robot.fairino import Robot
from robot.fairino_robot import FairinoRobot


# ----------------
# Global Config
# ----------------
ROBOT_IP = '192.168.58.2'
MODEL_NAME = 'FR5'
TOOL_ID = 1
USER_ID = 0
MOVE_VEL = 30.0
MOVE_ACC = 0.0
SETTLE_TIME = 1.0   # 到位后等待反馈稳定时间 (s)

TCP_RPY_ORDER = 'zyx'   # TCP 位姿欧拉角顺序

# 用于 "绝对精度" 测试的关节空间目标点 (deg)
TEST_JOINT_CONFIGS_DEG = [
    [  0.0,  -90.0,  90.0,  -90.0,  -90.0,   0.0],
    [ 20.0,  -70.0,  80.0, -100.0,  -60.0,  15.0],
    [-30.0,  -60.0,  60.0,  -90.0, -100.0, -20.0],
    [ 10.0, -100.0, 100.0,  -80.0,  -70.0,  30.0],
    [-15.0,  -80.0,  70.0, -110.0,  -90.0,  10.0],
]

# 用于 "重复定位精度" 测试的目标点 (deg)
REPEAT_TARGET_JOINT_DEG = [0.0, -90.0, 90.0, -90.0, -90.0, 0.0]
REPEAT_APPROACH_JOINT_DEG = [0.0, -70.0, 70.0, -70.0, -70.0, 0.0]

# Flange -> tool1 的固定偏移
T_FLANGE2TOOL1 = SE3(0.0, 0.0, 0.036)

# "定点旋转测试": 保持 tool1 原点不变, 绕其自身坐标轴叠加的姿态偏转
FIXED_POINT_RPY_OFFSETS_DEG = [
    (0, 0, 0),
    (15, 0, 0), (-15, 0, 0),
    (0, 15, 0), (0, -15, 0),
    (0, 0, 15), (0, 0, -15),
    (10, 10, 0), (-10, -10, 0),
]

# 定点旋转测试的基准姿态 (deg)，用于生成 tool1 中心点
FIXED_POINT_BASE_JOINT_DEG = [0, -90, 90, -90, -90, 0]


def tcp_pose_to_SE3(pose_mm_deg, order=TCP_RPY_ORDER):
    """把控制器反馈的 [x,y,z,rx,ry,rz] (mm, deg) 转成 SE3 (m)"""
    pose_mm_deg = np.asarray(pose_mm_deg, dtype=float)
    t = pose_mm_deg[:3] / 1000.0
    rpy = pose_mm_deg[3:]
    R = SE3.RPY(rpy, unit='deg', order=order)
    return SE3(t[0], t[1], t[2]) * R


def rotation_error_deg(T_model: SE3, T_actual: SE3) -> float:
    """计算两个位姿之间的旋转误差 (deg)"""
    R_err = T_model.R.T @ T_actual.R
    cos_theta = (np.trace(R_err) - 1.0) / 2.0
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_theta)))


def translation_error_mm(T_model: SE3, T_actual: SE3) -> float:
    """计算两个位姿之间的平移误差 (mm)"""
    return float(np.linalg.norm(T_model.t - T_actual.t) * 1000.0)


class FairinoAccuracyTester:
    """封装 运动学模型 + 实机 SDK 的精度 / 重复定位精度测试逻辑"""
    def __init__(self, ip: str = ROBOT_IP, dry_run: bool = True):
        self.dry_run = dry_run
        self.kinematics = FairinoRobot(model_name=MODEL_NAME)
        self.joint_limits = self.kinematics.get_joint_limits()
        print(f"[connect] 正在连接机械臂控制器 {ip} ...")
        self.robot = Robot.RPC(ip)
        print(f"[connect] 连接成功")


    # 基础状态读取 / 使能控制
    def get_actual_joint_rad(self) -> np.ndarray:
        err, joint_deg = self.robot.GetActualJointPosDegree()
        if err != 0:
            raise RuntimeError(f"读取当前关节位置失败.")
        return np.deg2rad(np.asarray(joint_deg, dtype=float))
    

    def get_actual_tcp_SE3(self) -> SE3:
        err, tcp = self.robot.GetActualTCPPose()
        if err != 0:
            raise RuntimeError(f"读取当前 TCP 位姿失败.")
        return tcp_pose_to_SE3(tcp)
    

    def enable(self):
        ret = self.robot.Mode(0)
        print(f"[enable] 切换自动模式.")
        time.sleep(0.5)
        ret = self.robot.RobotEnable(1)
        print(f"[enable] 上使能.")
        time.sleep(0.5)

    
    def disable(self):
        ret = self.robot.RobotEnable(0)
        print(f"[disable] 下使能.")


    def _check_limits(self, q_deg):
        q_rad = np.deg2rad(np.asarray(q_deg, dtype=float))
        for i, q in enumerate(q_rad):
            lo, hi = self.joint_limits[i]
            if not (lo <= q <= hi):
                print(f"[warning] J{i+1} 目标 {q_deg[i]:.1f} 超限位"
                      f"[{np.rad2deg(lo):.1f}, {np.rad2deg(hi):.1f}]")
                return False
        return True
    

    def _move_j_deg(self, q_deg, label=""):
        """下发 MoveJ (阻塞式), dry-run 模式下只做限位检查和模型计算"""
        if not self._check_limits(q_deg):
            return False
        
        if self.dry_run:
            q_rad = np.deg2rad(np.asarray(q_deg, dtype=float))
            pose_model = self.kinematics.forward_kinematics(q_rad)
            print(f"  [dry-run] {label} 目标 (deg): {np.round(q_deg, 3)}")
            print(f"  [dry-run] 模型位姿平移 (m): {np.round(pose_model.t, 4)}")
            return True
        
        ret = self.robot.MoveJ(list(q_deg), TOOL_ID, USER_ID, vel=MOVE_VEL, acc=MOVE_ACC, blendT=-1.0)
        if ret != 0:
            print(f"[warning] MoveJ 返回非零: {ret}")
        time.sleep(SETTLE_TIME)
        return True
    

    # ----------------------------------------
    # Test1: 绝对精度 
    # ----------------------------------------
    def test_accuracy(self, joint_configs_deg=None):
        """
        对一组关节空间目标点逐个运动, 比较:
            模型 FK vs 控制器反馈 TCP
        统计平移误差 (mm) / 旋转误差 (deg)
        """
        if joint_configs_deg is None:
            joint_configs_deg = TEST_JOINT_CONFIGS_DEG

        print("\n===== Test1: 多点绝对精度 =====")
        results = []

        for idx, q_target_deg in enumerate(joint_configs_deg):
            print(f"\n--- 点 {idx + 1} / {len(joint_configs_deg)} ---")
            ok = self._move_j_deg(q_target_deg, label=f"点 {idx + 1}")
            if not ok:
                continue
            if self.dry_run:
                continue

            q_actual = self.get_actual_joint_rad()
            pose_model_flange = self.kinematics.forward_kinematics(q_actual)
            # 控制器 GetActualTCPPose 返回的是 tool1 位姿,
            pose_model = pose_model_flange * T_FLANGE2TOOL1
            pose_actual = self.get_actual_tcp_SE3()

            trans_err = translation_error_mm(pose_model, pose_actual)
            rot_err = rotation_error_deg(pose_model, pose_actual)

            print(f"实际关节角 (deg): {np.round(np.rad2deg(q_actual), 3)}")
            print(f"[模型 FK, tool1]  平移 (m):  {np.round(pose_model.t, 4)}")
            print(f"[控制器]   平移 (m):  {np.round(pose_actual.t, 4)}")
            print(f"平移误差: {trans_err:.3f} mm | 旋转误差: {rot_err:.3f} deg")

            results.append({
                "point_index": idx,
                "target_joint_deg": list(q_target_deg),
                "actual_joint_deg": list(np.rad2deg(q_actual)),
                "translation_error_mm": trans_err,
                "rotation_error_deg": rot_err,
            })

        if not self.dry_run and results:
            trans_errs = [r["translation_error_mm"] for r in results]
            rot_errs = [r["rotation_error_deg"] for r in results]
            print("\n--- 绝对精度汇总 ---")
            print(f"平移误差: mean={np.mean(trans_errs):.3f} mm, "
                  f"max={np.max(trans_errs):.3f} mm, std={np.std(trans_errs):.3f} mm")
            print(f"旋转误差: mean={np.mean(rot_errs):.3f} deg, "
                  f"max={np.max(rot_errs):.3f} deg, std={np.std(rot_errs):.3f} deg")
        
        return results
    


    # -----------------------
    # Test2: 重复定位精度
    # -----------------------
    def test_repeatability(self, target_joint_deg=None, approach_joint_deg=None,
                            n_repeats: int = 10):
        """
        单向重复定位精度测试:
        每次先运动到"预备点" approach_joint_deg, 再运动到目标点 target_joint_deg,
        始终从同一方向逼近目标，排除反向间隙对结果的影响。
        重复 n_repeats 次，记录每次到位后控制器反馈的 TCP 位姿，
        用位置的离散程度 (ISO 9283 思路) 和旋转误差的 std/max 表示重复定位精度。
        """
        if target_joint_deg is None:
            target_joint_deg = REPEAT_TARGET_JOINT_DEG
        if approach_joint_deg is None:
            approach_joint_deg = REPEAT_APPROACH_JOINT_DEG
 
        print(f"\n===== 测试2: 重复定位精度 (单向逼近, 重复 {n_repeats} 次) =====")
 
        if self.dry_run:
            print("[dry-run] 该测试需要实际运动，请加 --execute 后运行")
            self._check_limits(target_joint_deg)
            self._check_limits(approach_joint_deg)
            return None
 
        positions_m = []
        rot_errs_vs_first = []
        pose_first = None
 
        for i in range(n_repeats):
            self._move_j_deg(approach_joint_deg, label=f"预备点(第{i + 1}次)")
            self._move_j_deg(target_joint_deg, label=f"目标点(第{i + 1}次)")
 
            pose_actual = self.get_actual_tcp_SE3()
            positions_m.append(pose_actual.t.copy())
 
            if pose_first is None:
                pose_first = pose_actual
            else:
                rot_errs_vs_first.append(rotation_error_deg(pose_first, pose_actual))
 
            print(f"第 {i + 1:2d} 次到位, 平移 (m): {np.round(pose_actual.t, 5)}")
 
        positions_m = np.array(positions_m)
        p_mean = positions_m.mean(axis=0)
        dists_mm = np.linalg.norm(positions_m - p_mean, axis=1) * 1000.0
 
        # ISO 9283 单向位置重复定位精度: RPl = mean(dist) + 3*std(dist)
        RPl_mm = float(np.mean(dists_mm) + 3.0 * np.std(dists_mm))
 
        print("\n--- 重复定位精度汇总 ---")
        print(f"位置分散度: mean={np.mean(dists_mm):.4f} mm, "
              f"max={np.max(dists_mm):.4f} mm, std={np.std(dists_mm):.4f} mm")
        print(f"位置重复定位精度 RPl (mean+3σ, 参考 ISO 9283): {RPl_mm:.4f} mm")
        if rot_errs_vs_first:
            print(f"姿态一致性 (相对第1次): mean={np.mean(rot_errs_vs_first):.4f} deg, "
                  f"max={np.max(rot_errs_vs_first):.4f} deg")
 
        return {
            "n_repeats": n_repeats,
            "positions_m": positions_m.tolist(),
            "dist_from_mean_mm": dists_mm.tolist(),
            "RPl_mm": RPl_mm,
            "rotation_consistency_deg": rot_errs_vs_first,
        }

    # ---------------------
    # Test3: 定点旋转 (保持 tool1 原点不变, 绕其自身轴转动)
    # ---------------------

    def test_fixed_point_rotation(self, base_joint_deg=None, rpy_offsets_deg=None):
        """
        保持 tool1 坐标系原点(中心点)不变，在该点上叠加一系列绕 tool1 自身坐标轴的
        姿态偏转 (roll/pitch/yaw)，通过逆运动学求解对应的关节角并运动过去，
        用于测试机械臂"定点旋转"的能力 —— 即姿态变化时，工具中心点实际漂移了多少。

        原理: 若 T_tool1_base 是基准 tool1 位姿，绕自身轴的姿态偏转应右乘:
              T_tool1_target = T_tool1_base * SE3.RPY(offset)
        因为 SE3.RPY(offset) 的平移分量为 0，右乘不会改变 T_tool1_target 的平移，
        即中心点在"指令"层面严格不变；实际漂了多少就是本测试要衡量的量。

        ⚠️ 依赖 self.kinematics 提供逆运动学接口。下面假设方法名为
        `inverse_kinematics(T_flange, q0=...)`，返回关节角 (rad) 或 None (无解)。
        如果你的 FairinoRobot 实现的方法名/返回值格式不同，请对应修改
        `_solve_ik` 里的调用方式。
        """
        if base_joint_deg is None:
            base_joint_deg = FIXED_POINT_BASE_JOINT_DEG
        if rpy_offsets_deg is None:
            rpy_offsets_deg = FIXED_POINT_RPY_OFFSETS_DEG

        print(f"\n===== 测试3: 定点旋转 (共 {len(rpy_offsets_deg)} 组姿态) =====")

        # 1) 建立基准 tool1 位姿：优先用实际反馈的关节角(若已运动到位)，
        #    dry-run 时直接用给定的基准关节角做模型计算。
        ok = self._move_j_deg(base_joint_deg, label="定点旋转-基准点")
        if not ok:
            print("基准点超限位，测试终止")
            return None

        if self.dry_run:
            q_seed = np.deg2rad(np.array(base_joint_deg, dtype=float))
        else:
            q_seed = self.get_actual_joint_rad()

        pose_flange_base = self.kinematics.forward_kinematics(q_seed)
        pose_tool1_base = pose_flange_base * T_FLANGE2TOOL1
        center_target_m = pose_tool1_base.t.copy()
        print(f"基准 tool1 中心点 (m): {np.round(center_target_m, 5)}")

        results = []
        actual_positions_m = []

        for idx, (roll, pitch, yaw) in enumerate(rpy_offsets_deg):
            label = f"定点旋转 第{idx + 1}/{len(rpy_offsets_deg)} 组 (r={roll},p={pitch},y={yaw})"
            print(f"\n--- {label} ---")

            T_offset = SE3.RPY([roll, pitch, yaw], unit="deg", order=TCP_RPY_ORDER)
            pose_tool1_target = pose_tool1_base * T_offset
            pose_flange_target = pose_tool1_target * T_FLANGE2TOOL1.inv()

            q_sol = self._solve_ik(pose_flange_target, q0=q_seed)
            if q_sol is None:
                print("  [警告] 逆运动学无解，跳过该姿态")
                continue

            q_sol_deg = np.rad2deg(q_sol)
            if not self._check_limits(q_sol_deg):
                continue

            if self.dry_run:
                print(f"  [dry-run] 目标关节角 (deg): {np.round(q_sol_deg, 2)}")
                print(f"  [dry-run] 目标 tool1 平移 (m, 应与基准一致): "
                      f"{np.round(pose_tool1_target.t, 5)}")
                continue

            ret = self.robot.MoveJ(list(q_sol_deg), TOOL_ID, USER_ID,
                                    vel=MOVE_VEL, acc=MOVE_ACC, blendT=-1.0)
            if ret != 0:
                print(f"  [警告] MoveJ 返回非零码: {ret}")
            time.sleep(SETTLE_TIME)

            pose_actual = self.get_actual_tcp_SE3()
            actual_positions_m.append(pose_actual.t.copy())

            drift_mm = translation_error_mm(
                SE3(center_target_m[0], center_target_m[1], center_target_m[2]),
                SE3(pose_actual.t[0], pose_actual.t[1], pose_actual.t[2]),
            )
            rot_track_err = rotation_error_deg(pose_tool1_target, pose_actual)

            print(f"  实际 tool1 平移 (m): {np.round(pose_actual.t, 5)}")
            print(f"  中心点漂移: {drift_mm:.4f} mm | 姿态跟踪误差: {rot_track_err:.4f} deg")

            results.append({
                "rpy_offset_deg": [roll, pitch, yaw],
                "joint_deg": q_sol_deg.tolist(),
                "actual_tool1_pos_m": pose_actual.t.tolist(),
                "center_drift_mm": drift_mm,
                "orientation_tracking_error_deg": rot_track_err,
            })

            q_seed = q_sol  # 用上一次解作为下一次 IK 的种子，帮助收敛到相邻解

        if not self.dry_run and actual_positions_m:
            actual_positions_m = np.array(actual_positions_m)
            centroid = actual_positions_m.mean(axis=0)
            dists_mm = np.linalg.norm(actual_positions_m - centroid, axis=1) * 1000.0
            drifts_mm = [r["center_drift_mm"] for r in results]
            rot_errs = [r["orientation_tracking_error_deg"] for r in results]

            print("\n--- 定点旋转汇总 ---")
            print(f"相对基准点的中心漂移: mean={np.mean(drifts_mm):.4f} mm, "
                  f"max={np.max(drifts_mm):.4f} mm, std={np.std(drifts_mm):.4f} mm")
            print(f"相对各姿态点均值的分散度 (定点球半径参考): "
                  f"mean={np.mean(dists_mm):.4f} mm, max={np.max(dists_mm):.4f} mm")
            print(f"姿态跟踪误差: mean={np.mean(rot_errs):.4f} deg, max={np.max(rot_errs):.4f} deg")

        return results

    def _solve_ik(self, T_flange_target: SE3, q0=None):
        """
        逆运动学求解包装函数，对齐 FairinoRobot.inverse_kinematics 的实际签名:
            inverse_kinematics(target_pose, reference_joint_pos=None) -> JointPos | None
        """
        q_sol = self.kinematics.inverse_kinematics(
            T_flange_target, reference_joint_pos=q0)
        if q_sol is None:
            return None
        return np.asarray(q_sol, dtype=float)


def main():
    parser = argparse.ArgumentParser(description="FR5 精度 / 重复定位精度测试脚本")
    parser.add_argument("--ip", default=ROBOT_IP, help="机械臂控制器 IP")
    parser.add_argument("--execute", action="store_true",
                         help="关闭 dry-run, 实际下发运动指令 (请确保现场安全后再使用)")
    parser.add_argument("--skip-accuracy", action="store_true", help="跳过绝对精度测试")
    parser.add_argument("--skip-repeatability", action="store_true", help="跳过重复定位精度测试")
    parser.add_argument("--skip-fixed-point", action="store_true", help="跳过定点旋转测试")
    parser.add_argument("--repeats", type=int, default=10, help="重复定位测试的重复次数")
    parser.add_argument("--save-json", default=None,
                         help="将测试结果保存为 JSON 文件的路径 (可选)")
    args = parser.parse_args()

    dry_run = not args.execute
    tester = FairinoAccuracyTester(ip=args.ip, dry_run=dry_run)

    all_results = {}

    try:
        if not dry_run:
            confirm = input(
                "\n即将下发实际运动指令进行精度测试, 确认机械臂周围安全、已准备好急停开关? (yes/no): ")
            if confirm.strip().lower() != "yes":
                print("用户取消，退出")
                return
            tester.enable()

        if not args.skip_accuracy:
            all_results["accuracy"] = tester.test_accuracy()

        if not args.skip_repeatability:
            all_results["repeatability"] = tester.test_repeatability(
                n_repeats=args.repeats)

        if not args.skip_fixed_point:
            all_results["fixed_point_rotation"] = tester.test_fixed_point_rotation()

        # 测试结束回到一个安全姿态
        if not dry_run:
            print("\n===== 回到安全位姿 =====")
            tester._move_j_deg(REPEAT_TARGET_JOINT_DEG, label="收尾")

    except Exception as e:
        print(f"\n[异常] {e}", file=sys.stderr)
    finally:
        if not dry_run:
            tester.disable()

    if args.save_json and all_results:
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        print(f"\n结果已保存到: {args.save_json}")


if __name__ == "__main__":
    main()