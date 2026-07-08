"""
FR5 实机测试脚本
=================

结合两部分能力：
1. `fairino_robot.py` 中的 `FairinoRobot` —— 纯运动学模型 (基于 roboticstoolbox),
   负责正/逆运动学计算和关节空间轨迹规划 (LSPB)。
2. 官方 `fairino` Python SDK —— 通过 RPC 与真实机械臂控制器通讯，
   负责读取实际状态、下发运动指令。

安装依赖:
    pip install fairino roboticstoolbox-python spatialmath-python numpy

    fairino SDK 官方仓库: https://github.com/FAIR-INNOVATION/fairino-python-sdk
    (不同固件版本的 SDK 接口可能略有差异，使用前建议核对随机械臂附带的 SDK 文档版本号)

⚠️ 安全须知 (务必先读):
    1. 首次运行务必使用默认的 dry-run 模式 (不加 --execute),
       仅测试连接、读取状态、正运动学校验，不会给机械臂下发任何运动指令。
    2. 确认 dry-run 结果符合预期后，再确保机械臂周围无障碍物、无人员在工作空间内，
       并保证急停开关在可触及范围内的情况下，加 --execute 参数执行实际运动测试。
    3. 脚本默认使用较低速度 (MOVE_VEL) 和很小的关节增量 (5°) 做验证性动作，
       正式使用前请结合你的现场情况调整速度、加速度和运动范围。
    4. 本脚本仅作为示例/测试用途，正式产线部署前需要完整的安全评审 (SIL/风险评估等)。
"""

import argparse
import sys
import time

import numpy as np

from spatialmath import SE3
from robot.fairino import Robot  # 官方 fairino Python SDK
from robot.fairino_robot import FairinoRobot  # 复用现有运动学模型 (FK / IK / 轨迹规划)


# --------------------
# 全局配置 (按需修改)
# --------------------
ROBOT_IP = "192.168.58.2"   # 机械臂控制器默认 IP，按现场配置修改
MODEL_NAME = "FR5"
TOOL_ID = 1                 # 工具坐标系编号 [0~14]
USER_ID = 0                 # 工件/参考坐标系编号 [0~14]
MOVE_VEL = 10.0              # MoveJ 速度百分比 [0~100]，测试阶段建议保持较低
MOVE_ACC = 0.0               # MoveJ 加速度百分比，SDK 暂未开放，保留默认值


def deg2rad(v):
    return np.deg2rad(v)


def rad2deg(v):
    return np.rad2deg(v)


class FairinoHardwareTester:
    """封装 运动学模型 + 实机 SDK 的联合测试逻辑"""

    def __init__(self, ip: str = ROBOT_IP, dry_run: bool = True):
        self.dry_run = dry_run

        # 运动学模型 (不连接硬件)
        self.kinematics = FairinoRobot(model_name=MODEL_NAME)
        # get_joint_limits() 返回 shape (dof, 2)：每一行是该关节的 [min, max] (rad)
        self.joint_limits = self.kinematics.get_joint_limits()

        # 实机连接
        print(f"[connect] 正在连接机械臂控制器 {ip} ...")
        self.robot = Robot.RPC(ip)
        print("[connect] 连接成功")

    # ---------------------
    # 基础状态读取 / 使能控制
    # ---------------------

    def get_actual_joint_rad(self) -> np.ndarray:
        """读取控制器反馈的实际关节角 (deg -> rad)"""
        err, joint_deg = self.robot.GetActualJointPosDegree()
        if err != 0:
            raise RuntimeError(f"读取当前关节位置失败, 错误码: {err}")
        return deg2rad(np.array(joint_deg, dtype=float))

    def enable(self):
        """切自动模式并上使能"""
        ret = self.robot.Mode(0)  # 0 - 自动运行模式
        print(f"[enable] 切换自动模式, ret={ret}")
        time.sleep(0.5)
        ret = self.robot.RobotEnable(1)  # 1 - 上使能
        print(f"[enable] 上使能, ret={ret}")
        time.sleep(0.5)

    def disable(self):
        """下使能 (测试结束/异常时调用，保证机械臂进入安全状态)"""
        ret = self.robot.RobotEnable(0)
        print(f"[disable] 下使能, ret={ret}")

    # ---------------------
    # 测试用例
    # ---------------------
 
    def test_connection_and_fk(self):
        """测试1: 读取实际关节角，并与运动学模型的正运动学结果做对比"""
        print("\n===== 测试1: 连接 & 正运动学校验 =====")
        q_actual = self.get_actual_joint_rad()
        print(f"实际关节角 (rad): {np.round(q_actual, 4)}")
        print(f"实际关节角 (deg): {np.round(rad2deg(q_actual), 2)}")
 
        pose_model = self.kinematics.forward_kinematics(q_actual)
        print(f"[模型 FK] 法兰盘位姿 (tool0):\n{pose_model}")
        print(f"[模型 FK] 法兰平移: {np.round(pose_model.t * 1000, 3)}")
        print(f"[模型 FK] 法兰旋转: {np.round(pose_model.rpy(unit='deg', order='zyx'), 3)}")

        T_flange2tool1 = SE3(0, 0, 0.036)
        pose_tool1 = pose_model * T_flange2tool1
        print(f"[模型 tool1] 位姿:\n{pose_tool1}")
        print(f"[模型 FK] tool1平移: {np.round(pose_tool1.t * 1000, 3)}")
        print(f"[模型 FK] 法兰旋转: {np.round(pose_tool1.rpy(unit='deg', order='zyx'), 3)}")
        # GetActualToolFlangePose 返回的是纯法兰盘位姿 (不含工具偏移),
        # 与 forward_kinematics 计算的是同一个坐标系，可以直接数值对比。
        try:
            err, tool1 = self.robot.GetActualTCPPose()
            if err == 0:
                print(f"[控制器反馈] TCP 位姿 (mm/deg): {np.round(tool1, 3)}")
            else:
                print(f"[控制器反馈] 读取失败, 错误码: {err}")
        except AttributeError:
            print("[控制器反馈] 当前 SDK 版本未找到 GetActualToolFlangePose 接口，"
                  "请用 dir(self.robot) 核对可用方法名")
 
        # 如果还想看当前 TOOL_ID 对应的真实 TCP 位姿 (含工具偏移)，可以顺便打一下作对照
        try:
            err, tool1 = self.robot.GetActualTCPPose()
            if err == 0:
                print(f"[控制器反馈] TCP 位姿 (含工具偏移, mm/deg): {np.round(tool1, 3)}")
        except AttributeError:
            pass
 
    def test_small_joint_move(self, joint_index: int = 5, delta_deg: float = 5.0):
        """测试2: 单关节小幅度运动 (自有轨迹规划 + 实机 MoveJ 执行)"""
        print(f"\n===== 测试2: 关节 J{joint_index + 1} 小幅度运动 (+{delta_deg}°) =====")
        q_start = self.get_actual_joint_rad()
        q_target = q_start.copy()
        q_target[joint_index] += deg2rad(delta_deg)
 
        lo, hi = self.joint_limits[joint_index][0], self.joint_limits[joint_index][1]
        if not (lo <= q_target[joint_index] <= hi):
            print(f"目标角度超出限位 [{rad2deg(lo):.1f}, {rad2deg(hi):.1f}]°, 已跳过")
            return
 
        # 复用现有的关节空间轨迹规划 (仅用于验证规划结果，实际下发用 MoveJ 走终点)
        trajectory = self.kinematics.move_joint_space(
            q_start, q_target, velocity=0.3, acceleration=0.5)
        if trajectory is None:
            print("轨迹规划失败")
            return
        print(f"规划轨迹点数: {len(trajectory)}")
 
        target_deg = list(rad2deg(q_target))
        print(f"目标关节角 (deg): {np.round(target_deg, 2)}")
 
        if self.dry_run:
            print("[dry-run] 跳过实际下发指令")
            return
 
        # blendT=-1.0 表示阻塞式运动到位
        ret = self.robot.MoveJ(target_deg, TOOL_ID, USER_ID,
                                vel=MOVE_VEL, acc=MOVE_ACC, blendT=-1.0)
        print(f"MoveJ 返回码: {ret}")
 
        time.sleep(0.5)
        q_after = self.get_actual_joint_rad()
        print(f"运动后实际关节角 (deg): {np.round(rad2deg(q_after), 2)}")
 
    def test_return_home(self, q_home_deg=None):
        """测试3: 返回预设安全位姿"""
        print("\n===== 测试3: 返回安全位姿 =====")
        if q_home_deg is None:
            q_home_deg = [0, -90, 90, -90, -90, 0]  # 请按你机台实际情况修改为安全位姿
        print(f"目标关节角 (deg): {q_home_deg}")
 
        if self.dry_run:
            print("[dry-run] 跳过实际下发指令")
            return
 
        ret = self.robot.MoveJ(q_home_deg, TOOL_ID, USER_ID,
                                vel=MOVE_VEL, acc=MOVE_ACC, blendT=-1.0)
        print(f"MoveJ 返回码: {ret}")
 
    def diagnose_joint_sign_convention(self, test_angle_deg: float = 30.0,
                                        settle_time: float = 1.0):
        """
        诊断用: 从全零位开始，每次只转一个关节到 test_angle_deg,
        对比模型 FK 与控制器法兰盘位姿，用于定位哪个关节的
        旋转方向/零点约定与实机不一致。
 
        ⚠️ 会实际驱动机械臂逐个关节运动，必须在 --execute 模式下才会真正执行，
        运行前请确保机械臂周围有足够安全空间做全关节范围运动。
        """
        print(f"\n===== 诊断: 单关节扫描 (每次转动 {test_angle_deg}°) =====")
        dof = len(self.joint_limits)
        q_zero = np.zeros(dof)
 
        if self.dry_run:
            print("[dry-run] 该诊断需要实际运动，请加 --execute 后配合 --diagnose 使用")
            return
 
        ret = self.robot.MoveJ(list(rad2deg(q_zero)), TOOL_ID, USER_ID,
                                vel=MOVE_VEL, acc=MOVE_ACC, blendT=-1.0)
        print(f"回零位, MoveJ 返回码: {ret}")
        time.sleep(settle_time)
 
        for i in range(dof):
            q_test = q_zero.copy()
            q_test[i] = deg2rad(test_angle_deg)
 
            lo, hi = self.joint_limits[i][0], self.joint_limits[i][1]
            if not (lo <= q_test[i] <= hi):
                print(f"J{i + 1}: 目标角度超出限位, 跳过")
                continue
 
            ret_move = self.robot.MoveJ(list(rad2deg(q_test)), TOOL_ID, USER_ID,
                                         vel=MOVE_VEL, acc=MOVE_ACC, blendT=-1.0)
            time.sleep(settle_time)
 
            # 直接核对运动后的实际关节角，与目标角度做比较，排除"命令没执行成功"这一可能性
            q_readback = self.get_actual_joint_rad()
            joint_error_deg = rad2deg(q_readback) - rad2deg(q_test)
 
            pose_model = self.kinematics.forward_kinematics(q_test)
            err, tool0_pose = self.robot.GetActualTCPPose()
 
            print(f"\n--- J{i + 1} = {test_angle_deg}°, 其余关节 = 0° ---")
            print(f"MoveJ 返回码: {ret_move}")
            print(f"实际关节角回读 (deg): {np.round(rad2deg(q_readback), 3)} "
                  f"(与目标误差: {np.round(joint_error_deg, 3)})")
            print(f"[模型 FK] 平移 (m): {np.round(pose_model.t, 4)}")
            if err == 0:
                print(f"[控制器] Tool0位姿 (mm/deg): {np.round(tool0_pose, 3)}")
            else:
                print(f"[控制器] 读取失败, 错误码: {err}")
 
            ret_back = self.robot.MoveJ(list(rad2deg(q_zero)), TOOL_ID, USER_ID,
                                         vel=MOVE_VEL, acc=MOVE_ACC, blendT=-1.0)
            print(f"回零位, MoveJ 返回码: {ret_back}")
            time.sleep(settle_time)
 
        print("\n诊断完成: 请逐个关节比对模型 FK 和控制器法兰盘位姿，"
              "哪个关节的位移方向/角度符号明显不一致，就说明该关节的 "
              "theta_offset 或旋转方向与实机定义相反。")

def main():
    parser = argparse.ArgumentParser(description="FR5 实机测试脚本")
    parser.add_argument("--ip", default=ROBOT_IP, help="机械臂控制器 IP")
    parser.add_argument("--execute", action="store_true",
                         help="关闭 dry-run, 实际下发运动指令 (请确保现场安全后再使用)")
    parser.add_argument("--diagnose", action="store_true",
                         help="运行单关节扫描诊断 (需配合 --execute)，用于定位 FK 偏差来源")
    parser.add_argument("--diagnose-angle", type=float, default=30.0,
                         help="诊断时每个关节转动的角度 (deg)，默认 30")
    args = parser.parse_args()

    dry_run = not args.execute

    tester = FairinoHardwareTester(ip=args.ip, dry_run=dry_run)

    try:
        tester.test_connection_and_fk()

        if not dry_run:
            confirm = input(
                "\n即将下发实际运动指令, 确认机械臂周围安全、已准备好急停开关? (yes/no): ")
            if confirm.strip().lower() != "yes":
                print("用户取消，退出")
                return
            tester.enable()

        if args.diagnose:
            tester.diagnose_joint_sign_convention(test_angle_deg=args.diagnose_angle)
        else:
            tester.test_small_joint_move(joint_index=5, delta_deg=5.0)
            tester.test_return_home()

    except Exception as e:
        print(f"\n[异常] {e}", file=sys.stderr)
    finally:
        if not dry_run:
            tester.disable()


if __name__ == "__main__":
    main()
