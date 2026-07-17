import numpy as np
from typing import Optional, List
import roboticstoolbox as rtb
from .robot_base import RobotBase, JointPos, Pose, JointLimit, JointTrajectory, PoseTrajectory
from .fairino_dh_params import get_mdh_params, get_supported_models, DEFAULT_QLIM_RAD, FR3_C_QLIM_RAD


class FairinoRobot(RobotBase):
    """
    Fairino 机器人实现类, 基于 Robotics Toolbox for Python (RTB) 实现
    支持通过 model_name 选择不同型号 (如： FR5, FR3C) 的 MDH 参数化模型, 并提供正运动学、逆运动学、关节空间运动等功能
    型号与 MDH 参数的对应关系维护在 fairino_dh_params.py 中
    """

    def __init__(self, model_name: str = "FR5", joint_limits: Optional[List[JointLimit]] = None):
        """
        :param model_name: Fairino 型号名称, 大小写不敏感, 完整支持列表见 fairino_dh_params.get_supported_models()
        :param joint_limits: 可选的关节限位列表, 若未提供则使用默认的关节限位
        """
        super().__init__(model_name=model_name, degree_of_freedom=6)

        self.DEFAULT_JOINT_VEL_LIMIT = 1.0  # 默认关节速度限制(rad/s)
        self.DEFAULT_JOINT_ACC_LIMIT = 2.0  # 默认关节加速度限制(rad/s^2)
        self.TRAJECTORY_FREQUENCY = 50.0    # 轨迹规划频率(Hz)

        # 按型号加载 MDH 参数 (未收录型号会抛出 KeyError, 并提示可选型号)
        mdh_params = get_mdh_params(model_name)
        if model_name.strip().upper() == "FR3-C":
            qlim = joint_limits if joint_limits is not None else FR3_C_QLIM_RAD
        else:
            qlim = joint_limits if joint_limits is not None else DEFAULT_QLIM_RAD

        links = [
            rtb.RevoluteMDH(
                a=param.a,
                d=param.d,
                alpha=param.alpha,
                offset=param.theta_offset,
                qlim=list(qlim[i])
            )
            for i, param in enumerate(mdh_params)
        ]

        self._robot = rtb.DHRobot(links, name=model_name)
        self._joint_limits = self._robot.qlim.T
        self._q = np.zeros(self.get_dof())  # 当前关节位置初始化为零
        self._dq = np.zeros(self.get_dof())  # 当前关节速度初始化为零
        self._pose = self.forward_kinematics(self._q)  # 当前末端位姿初始化为零位姿

    # -------------
    # 工具方法
    # -------------

    @staticmethod
    def list_supported_models() -> List[str]:
        """
        获取当前已收录 MDH 参数的 Fairino 机器人型号列表
        :return: 支持的 Fairino 机器人型号列表
        """
        return get_supported_models()

    def get_joint_limits(self) -> List[JointLimit]:
        """
        获取当前机器人关节限位
        :return: 关节限位 (rad)
        """
        return self._joint_limits

    def check_joint_limits(self, joint_pos: JointPos, margin_rad: float = 0.0):
        """
        检查给定关节位置是否超出(或过于接近)关节限位。

        用途: ikine_LM 是纯数值迭代求解, 不会主动把解约束在 qlim 之内, 所以 IK
        算出来的解仍可能超出实际关节限位; 一旦把这种解通过 ServoJ 下发给控制器,
        控制器会报 "PTP指令关节超限" 之类的错误, 且机械臂会直接"挂掉"停止响应,
        必须去 WebApp 里手动复位才能继续。因此在下发前应主动做这一层检查, 提前
        拦截, 而不是等控制器报错。

        :param joint_pos: 待检查的关节位置 (rad)
        :param margin_rad: 安全裕度 (rad), 正值表示比真实限位更早触发告警(留出
                            缓冲区, 避免刚好卡在硬限位上, 更保守), 默认0表示只按
                            真实限位判断
        :return: (ok, violations)
                 ok: bool, 全部关节都在(收紧后的)限位内时为 True
                 violations: List[(joint_index, q_rad, qmin_rad, qmax_rad)],
                             每一项对应一个超限的关节, joint_index 从0开始
        """
        q = np.asarray(joint_pos, dtype=float)
        violations = []
        for i, (qmin, qmax) in enumerate(self._joint_limits):
            lo = qmin + margin_rad
            hi = qmax - margin_rad
            if q[i] < lo or q[i] > hi:
                violations.append((i, float(q[i]), float(qmin), float(qmax)))
        return len(violations) == 0, violations

    # -------------
    # 运动学方法
    # -------------

    def forward_kinematics(self, joint_pos: JointPos) -> Pose:
        """
        计算正运动学, 根据关节位置计算末端位姿
        :param joint_pos: 关节位置 (rad)
        :return: 末端位姿 (SE3)
        """
        q = np.asarray(joint_pos)
        pose = self._robot.fkine(q)
        return pose

    def inverse_kinematics(self, target_pose: Pose,
                           reference_joint_pos: Optional[JointPos] = None) -> Optional[JointPos]:
        """
        逆运动学计算, 根据目标位姿计算关节位置
        :params target_pose: 目标位姿 (空间坐标系)
        :params reference_joint_pos: 参考关节位置 (rad), 作为数值迭代初值, 不传则使用机器人当前关节位置
        :return: 关节位置 (rad), 无解时返回 None
        """
        # 如果提供了参考位置则使用, 否则使用机器人当前记录的位置
        q0 = np.asarray(
            reference_joint_pos) if reference_joint_pos is not None else self._q
        # 使用 Levenberg-Marquardt (LM) 数值算法求解
        sol = self._robot.ikine_LM(target_pose, q0=q0)
        if not sol.success:
            return None
        return sol.q

    # ------------
    # 运动规划
    # ------------

    def move_joint_space(self,
                         current_joint_pos: JointPos,
                         target_joint_pos: JointPos,
                         velocity: Optional[float] = None,
                         acceleration: Optional[float] = None) -> Optional[JointTrajectory]:
        """
        关节空间运动: 使用梯形速度规划器 (LSPB) 规划同步轨迹
        :params current_joint_pos: 当前关节位置 (rad)
        :params target_joint_pos: 目标关节位置 (rad)
        :params velocity: 关节运动速度 (rad/s)
        :params acceleration: 关节运动加速度 (rad/s^2)
        :return: 关节空间轨迹 (每个采样点为一组关节角)
        """
        # 1. 数据准备与校验
        q_start = np.array(current_joint_pos, dtype=float)
        q_end = np.array(target_joint_pos, dtype=float)
        dof = self.get_dof()

        if q_start.shape != (dof,) or q_end.shape != (dof,):
            print(f"Error: Joint dim mismatch.")
            return None

        if np.allclose(q_start, q_end, atol=1e-5):
            return [q_start, q_end]

        # 获取速度和加速度限制标量
        v_limit = max(
            velocity if velocity is not None else self.DEFAULT_JOINT_VEL_LIMIT, 1e-4)
        a_limit = max(
            acceleration if acceleration is not None else self.DEFAULT_JOINT_ACC_LIMIT, 1e-4)

        # 2. 计算同步所需最短总时间 T_total
        # 这个时间保证了最慢的关节也能在 v_limit 和 a_limit 内完成运动
        max_duration = 0.0
        threshold_dist = v_limit**2 / a_limit

        for i in range(dof):
            dist = abs(q_end[i] - q_start[i])
            if dist < 1e-6:
                continue

            if dist > threshold_dist:
                # 梯形分布 (Trapezoidal)
                duration_i = dist / v_limit + v_limit / a_limit
            else:
                # 三角形分布 (Triangular)
                duration_i = 2 * np.sqrt(dist / a_limit)

            if duration_i > max_duration:
                max_duration = duration_i

        # 确保最小持续时间, 避免步数过少
        T_total = max(max_duration, 2.0 / self.TRAJECTORY_FREQUENCY)

        # 3. 手动生成 LSPB 轨迹
        # 计算总步数和时间向量
        steps = int(np.ceil(T_total * self.TRAJECTORY_FREQUENCY))
        t_vec = np.linspace(0, T_total, steps)

        # 初始化轨迹数组 [steps x dof]
        full_trajectory_np = np.zeros((steps, dof))

        for i in range(dof):
            dist = q_end[i] - q_start[i]
            dist_abs = abs(dist)
            sign = np.sign(dist)

            if dist_abs < 1e-6:
                full_trajectory_np[:, i] = q_start[i]
                continue

            # 为了同步，强制所有关节使用总时间 T_total, 并固定使用最大加速度 a_limit
            # 需要计算在此约束下，该关节所需要的峰值速度 v_peak_i
            # 基于 LSPB 时间公式 T = D/v + v/a, 解关于 v 的二次方程: v^2 - (Ta)v + Da = 0
            # 取较小的根作为峰值速度, 以确保它是满足条件的最小必要速度

            # 判别式 delta = (Ta)^2 - 4Da
            delta = (T_total * a_limit)**2 - 4 * dist_abs * a_limit
            # 由于 T_total 是基于最慢关节计算的, 理论上 delta >= 0, 取 abs 防止微小数值误差
            v_peak_i = (T_total * a_limit - np.sqrt(abs(delta))) / 2.0

            # 计算该关节的加速时间
            t_acc_i = v_peak_i / a_limit
            # 计算加速阶段走过的距离
            dist_acc_i = 0.5 * a_limit * t_acc_i**2

            # 在每个时间步计算位置
            for j in range(steps):
                t = t_vec[j]
                if t <= t_acc_i:
                    # 加速段: q = q0 + sign * 0.5 * a * t^2
                    q_t = q_start[i] + sign * 0.5 * a_limit * t**2
                elif t <= T_total - t_acc_i:
                    # 匀速段: q = q0 + sign * d_acc + sign * v * (t - t_acc)
                    q_t = q_start[i] + sign * \
                        (dist_acc_i + v_peak_i * (t - t_acc_i))
                else:
                    # 减速段: q = qf - sign * 0.5 * a * (T - t)^2
                    t_rem = T_total - t
                    q_t = q_end[i] - sign * 0.5 * a_limit * t_rem**2

                full_trajectory_np[j, i] = q_t

        # 强制最后一个点为目标点, 消除累积误差
        full_trajectory_np[-1, :] = q_end

        # 4. 返回结果
        return list(full_trajectory_np)

    def move_to_pose(self,
                     target_pose: Pose,
                     reference_joint_pos: Optional[JointPos] = None) -> Optional[PoseTrajectory]:
        """
        规划一条从当前位置到目标位置的点到点 (PTP) 轨迹
        """
        q_start = np.array(reference_joint_pos)
        q_end = self.inverse_kinematics(
            target_pose, reference_joint_pos=q_start)

        if q_end is None:
            print(
                f"Error: [move_to_pose] IK solver failed to find a solution for the target pose.")
            return None

        joint_trajectory = self.move_joint_space(current_joint_pos=q_start,
                                                 target_joint_pos=q_end)
        
        if joint_trajectory is None:
            print(f"Error: [move_to_pose] Joint space trajectory planning failed.")
            return None
        
        # 遍历生成的每一个关节配置点, 通过正运动学 (FK) 计算出对应的末端位姿
        pose_trajectory: PoseTrajectory = []
        for q in joint_trajectory:
            pose = self.forward_kinematics(q)
            pose_trajectory.append(pose)

        return pose_trajectory