"""
scripts/algorithm/Pbvs.py

基于位置的视觉伺服 (Position-Based Visual Servoing, PBVS) 核心算法模块

本.py的底层控制律(PoseServoLaw)实现思路:
    - 用 SO3.interp (插值/slerp) 来计算增量姿态,
      且通过 fix_rotation_matrix 的 SVD 正交化对相机噪声/累积
      浮点误差更鲁棒(旋转矩阵在多次相乘后可能不再严格正交, SVD 修正可以避免
      IK/后续插值因非正交矩阵而出现异常)。
    - 比例增益按"单步最大平移/旋转量"限幅, 每个控制周期调用一次 step(), 
      增益/步长的物理意义是"每次调用走多远", 因此和调用频率(--servo-hz) 是绑定的, 调参时需要一并考虑。
    - 增加了工作空间限幅(workspace_min/max, base 坐标系下), 作为一层额外的
      硬件安全保护, 防止 marker 检测异常/外参错误导致目标位姿飞出安全范围。
    - step() 返回 WORKING/FINISHED/ERROR_INVALID_INPUT 状态码, 便于上层判断
      "已收敛到当前目标"还是"数据非法(NaN/Inf)需要跳过"。
 
与 visual_servo_hover_test.py "算一次目标位姿 -> IK 一次 -> MoveJ 阻塞式到位"
的做法不同, PBVS 是逐帧闭环控制: 每个控制周期都根据当前误差走一小步, 因此
即使目标(标定板)持续运动, 也能连续跟随, 而不是重新规划一条到静止点的轨迹。
 
本模块只依赖 numpy / spatialmath, 不涉及任何机器人 SDK 或相机接口,
便于离线单元测试; 与硬件/相机的绑定放在 main_pbvs.py 中完成。
"""

from collections import deque
from dataclasses import dataclass, field
import time
from typing import Deque, Optional, Tuple

import numpy as np
from spatialmath import SE3, SO3


# =============
# 数值鲁棒性工具
# =============

def fix_rotation_matrix(R: np.ndarray) -> np.ndarray:
    """
    修正旋转矩阵使其满足 SO(3) 约束(正交 + 行列式为+1)。
    多次位姿相乘(尤其是相机观测经手眼外参多级换算后)容易因浮点误差导致
    旋转矩阵轻微失去正交性; 直接喂给 SO3()/IK 可能报错或产生异常结果。
    这里用 SVD 分解重新投影到最近的合法旋转矩阵上。
    :param R: 3x3 旋转矩阵 (可能因浮点误差不严格正交)
    :return: 修正后的正交旋转矩阵 (det = +1)
    """
    U, _, Vh = np.linalg.svd(R)
    R_fixed = U @ Vh
    if np.linalg.det(R_fixed) < 0:
        U[:, -1] *= -1
        R_fixed = U @ Vh
    return R_fixed


def _to_safe_SE3(T: SE3) -> SE3:
    """把 T.R 修正为合法旋转矩阵后, 重新组装成 SE3"""
    return SE3.Rt(fix_rotation_matrix(T.R), T.t, check=False)


def pose_error_magnitudes(T_a: SE3, T_b: SE3) -> Tuple[float, float]:
    """
    计算两个位姿之间的误差大小: (位置误差模长[m], 姿态误差角度[rad])。
    内部对误差旋转矩阵做 SVD 正交化修正, 避免 SO3().angvec() 因非正交矩阵报错。 
    :param T_a: 位姿 A (如: 当前位姿)
    :param T_b: 位姿 B (如: 期望位姿)
    :return: (position_error_m, rotation_error_rad)
    """
    error_pose = _to_safe_SE3(T_a.inv() * T_b)
    position_error = float(np.linalg.norm(error_pose.t))
    rotation_error = float(SO3(error_pose.R).angvec()[0])
    return position_error, rotation_error


def interp_pose(current: SE3, target: SE3, s_position: float, s_orientation: float) -> SE3:
    """
    在 current 与 target 之间插值: 平移线性插值, 旋转用 SO3.interp (球面插值)。
    :param s_position: 平移插值系数, 0=current, 1=target
    :param s_orientation: 旋转插值系数, 0=current, 1=target
    """
    out_translation = (1 - s_position) * current.t + s_position * target.t
    R_current = SO3(fix_rotation_matrix(current.R))
    R_target = SO3(fix_rotation_matrix(target.R))
    out_R = R_current.interp(R_target, s_orientation)
    return SE3.Rt(out_R, out_translation, check=False)


def interp_pose_from_identity(end: SE3, s_position: float, s_orientation: float) -> SE3:
    """从单位位姿(Identity)插值到 end, 用于把"误差位姿"按比例缩小成"增量位姿" """
    return interp_pose(SE3(), end, s_position, s_orientation)


# ========================
# 底层控制律: 插值式比例控制
# ========================

class PoseServoLaw:
    """
    纯数学的位姿控制律: 输入(当前位姿, 期望位姿), 输出下一步应到达的位姿
    (每次调用前进一小步), 不依赖任何机器人模型, 也不依赖时间步长 dt——
    每次 step() 调用代表"一个控制周期", 步长限幅的物理意义是"每周期最多走
    多远", 因此与调用频率(main_pbvs.py 里的 --servo-hz)是绑定的。
    """

    RESULT_WORKING = 0
    RESULT_FINISHED = 1
    RESULT_ERROR_INVALID_INPUT = -1

    def __init__(
            self,
            kp: float = 0.3,
            max_translation_step: float = 0.01,               # 单步最大平移量 (m)
            max_rotation_step: float = np.deg2rad(5),         # 单步最大旋转角 (rad)
            # base 坐标系下工作空间下限 [x,y,z] (m), None=不限制
            workspace_min: Optional[np.ndarray] = None,
            # base 坐标系下工作空间上限 [x,y,z] (m), None=不限制
            workspace_max: Optional[np.ndarray] = None,
            position_error_threshold: float = 0.001,          # 位置收敛阈值 (m)
            rotation_error_threshold: float = np.deg2rad(0.5)  # 旋转收敛阈值 (rad)
    ):
        self.kp = float(kp)
        self.max_translation_step = float(max_translation_step)
        self.max_rotation_step = float(max_rotation_step)
        self.workspace_min = np.asarray(
            workspace_min, dtype=float) if workspace_min is not None else None
        self.workspace_max = np.asarray(
            workspace_max, dtype=float) if workspace_max is not None else None
        self.position_error_threshold = float(position_error_threshold)
        self.rotation_error_threshold = float(rotation_error_threshold)
        self.counter = 0

    def reset(self):
        self.counter = 0

    def _compute_increment(self, current_pose: SE3, desired_pose: SE3) -> Tuple[SE3, float, float]:
        """
        计算从 current_pose 走向 desired_pose 的增量位姿(在 current_pose 本体系下表达),
        并返回本次调用前的误差大小(position_error_m, rotation_error_rad), 供收敛判断
        和上层日志/静止检测复用, 避免重复计算。
        """
        error_pose = _to_safe_SE3(current_pose.inv() * desired_pose)
        position_error = float(np.linalg.norm(error_pose.t))
        rotation_error = float(SO3(error_pose.R).angvec()[0])

        kp_position = self.kp
        kp_orientation = self.kp
        eps = 1e-9

        if position_error > eps:
            if position_error * self.kp > self.max_translation_step:
                kp_position = self.max_translation_step / position_error
        else:
            kp_position = 0.0

        if rotation_error > eps:
            if rotation_error * self.kp > self.max_rotation_step:
                kp_orientation = self.max_rotation_step / rotation_error
        else:
            kp_orientation = 0.0

        pose_incr = interp_pose_from_identity(
            error_pose, kp_position, kp_orientation)
        return pose_incr, position_error, rotation_error

    def step(self, current_pose: SE3, desired_pose: SE3) -> Tuple[int, SE3, float, float]:
        """
        执行一步 PBVS 控制。
        :param current_pose: 当前位姿 (如实时 FK 得到的 flange 位姿)
        :param desired_pose: 期望位姿 (如由 marker 观测换算出的目标 flange 位姿)
        :return: (status, next_pose, position_error_m, rotation_error_rad)
                 status 取值: RESULT_WORKING / RESULT_FINISHED / RESULT_ERROR_INVALID_INPUT
                 non-ERROR 时 next_pose 是应下发的下一步目标位姿(已做工作空间限幅);
                 ERROR 时 next_pose 原样返回 current_pose, 上层应跳过本周期。
        """
        self.counter += 1

        if not np.all(np.isfinite(current_pose.A)) or not np.all(np.isfinite(desired_pose.A)):
            return self.RESULT_ERROR_INVALID_INPUT, current_pose, float('nan'), float('nan')

        pose_incr, position_error, rotation_error = self._compute_increment(
            current_pose, desired_pose)

        if (position_error < self.position_error_threshold and
                rotation_error < self.rotation_error_threshold):
            return self.RESULT_FINISHED, current_pose, position_error, rotation_error

        new_pose = current_pose * pose_incr

        if self.workspace_min is not None and self.workspace_max is not None:
            clamped_t = np.clip(
                new_pose.t, self.workspace_min, self.workspace_max)
            new_pose = SE3.Rt(new_pose.R, clamped_t, check=False)

        return self.RESULT_WORKING, new_pose, position_error, rotation_error


# 向后兼容别名: PBVSGains 不再是必需的独立参数容器(PoseServoLaw 构造函数直接
# 接收所有增益/限幅参数), 保留一个薄封装方便按名字传参/复用同一套参数。
@dataclass
class PBVSGains:
    kp: float = 0.3
    max_translation_step: float = 0.01
    max_rotation_step: float = np.deg2rad(5)
    workspace_min: Optional[np.ndarray] = None
    workspace_max: Optional[np.ndarray] = None
    position_error_threshold: float = 0.001
    rotation_error_threshold: float = np.deg2rad(0.5)

    def build(self) -> PoseServoLaw:
        return PoseServoLaw(
            kp=self.kp,
            max_translation_step=self.max_translation_step,
            max_rotation_step=self.max_rotation_step,
            workspace_min=self.workspace_min,
            workspace_max=self.workspace_max,
            position_error_threshold=self.position_error_threshold,
            rotation_error_threshold=self.rotation_error_threshold,
        )


# ======================================
# 高层控制器: 组合控制律 + 运动学模型(FK/IK)
# ======================================

class PBVSController:
    """
    组合 PoseServoLaw 与机器人运动学模型(FairinoRobot 或任何实现了
    inverse_kinematics(target_pose, reference_joint_pos) 接口的对象),
    提供"给定期望 flange 位姿 -> 输出下一步关节角目标"的高层接口。
    kinematics_model 只用到 inverse_kinematics 方法, 因此这里不做类型强绑定,
    只要求鸭子类型(duck typing)满足接口即可, 方便测试时传入 mock 对象。
    """

    def __init__(self, kinematics_model, gains: Optional[PBVSGains] = None,
                 servo_law: Optional[PoseServoLaw] = None):
        self.kinematics_model = kinematics_model
        self.law = servo_law or (
            gains.build() if gains is not None else PoseServoLaw())

    def compute_next_joint_target(
            self,
            base_T_flange_current: SE3,
            base_T_flange_desired: SE3,
            q_seed: Optional[np.ndarray] = None,
    ) -> Tuple[Optional[np.ndarray], SE3, int, float, float]:
        """
         计算下一步的关节角目标。

         :param base_T_flange_current: 当前 flange 位姿 (base 坐标系下, 一般来自实时 FK)
         :param base_T_flange_desired: 期望 flange 位姿 (base 坐标系下, 一般来自 marker 观测换算)
         :param q_seed: IK 数值迭代初值(建议传入当前关节角, 保证解连续、避免跳变)
         :return: (q_next, T_next, status, position_error_m, rotation_error_rad)
                  q_next 为 None 表示 status 是 ERROR, 或者 IK 无解, 上层应跳过本周期、
                  保持上一次目标不变
         """
        status, T_next, pos_err, rot_err = self.law.step(
            base_T_flange_current, base_T_flange_desired)
        if status == PoseServoLaw.RESULT_ERROR_INVALID_INPUT:
            return None, T_next, status, pos_err, rot_err

        q_next = self.kinematics_model.inverse_kinematics(
            T_next, reference_joint_pos=q_seed)
        return q_next, T_next, status, pos_err, rot_err


# ===========================================
# 悬停目标位姿计算 (Z 轴对齐, 朝向随 marker 偏转)
# ===========================================
def compute_hover_flange_pose(base_T_marker: SE3,
                              flange_T_ref: SE3,
                              hover_distance: float,
                              z_align: bool = True,) -> SE3:
    """
    计算"参考坐标系(如相机或 tool1)Z 轴与 marker Z 轴对齐/反向, 并悬停在
    marker 正上方 hover_distance 处"时, 所需的 flange 目标位姿。

    朝向随 marker 的姿态整体偏转(而不是固定朝向), 即: 参考坐标系的姿态
    始终等于 marker 姿态(z_align=True 时额外绕 X 轴翻转 180°, 实现"俯视"),
    这样当标定板倾斜时, 末端的朝向会跟着一起偏转。

    :param base_T_marker: marker 在 base 坐标系下的位姿 (来自 手眼外参 + 实时FK 换算)
    :param flange_T_ref: 参考坐标系(camera 或 tool1)相对 flange 的外参(手眼标定结果,
                          或 T_FLANGE_TO_TOOL1), 用于把目标位姿从"参考坐标系"换算回
                          flange 坐标系下发给 IK。
    :param hover_distance: 悬停距离 (m), 沿 marker Z 轴负方向(即参考坐标系 Z 轴反向)
    :param z_align: True 时参考坐标系 Z 轴与 marker Z 轴反向(俯视/正对标记);
                     False 时参考坐标系姿态与 marker 完全一致(只做 Z 方向平移)
    :return: base_T_flange_target (SE3)
    """
    if z_align:
        marker_T_target_frame = SE3.Tz(
            hover_distance) * SE3.Rx(180, unit='deg')
    else:
        marker_T_target_frame = SE3.Tz(hover_distance)

    base_T_target_frame = base_T_marker * marker_T_target_frame
    base_T_flange_target = base_T_target_frame * flange_T_ref.inv()
    return base_T_flange_target


# ====================================
# 位姿低通滤波 (可选, 用于平滑相机观测噪声)
# ====================================
class PoseExponentialFilter:
    """
    对 SE3 位姿序列做一阶低通滤波: 平移线性插值, 旋转用 SO3.interp 球面插值,
    与 PoseServoLaw 内部使用同一套 interp_pose_from_identity 逻辑, 避免单独
    对旋转矩阵做线性平均导致的非正交问题。

    alpha 越接近 1, 跟随越快(滤波越弱); 越接近 0, 越平滑(滞后越大)。

    位置和姿态可以设置不同的平滑系数(alpha_pos / alpha_rot): ArUco solvePnP
    对姿态(尤其俯仰/横滚轴)的噪声通常比对位置的噪声大得多, 而 z_align 模式
    又会把 marker 姿态直接映射成末端姿态, 姿态噪声会通过运动学被放大成明显
    的关节抖动。因此姿态建议用比位置更强的平滑(更小的 alpha_rot)。
    """

    def __init__(self, alpha: float = 0.4, alpha_rot: Optional[float] = None):
        if not (0.0 < alpha <= 1.0):
            raise ValueError("alpha 必须在 (0, 1] 之间")
        self.alpha_pos = alpha
        self.alpha_rot = alpha_rot if alpha_rot is not None else alpha
        if not (0.0 < self.alpha_rot <= 1.0):
            raise ValueError("alpha_rot 必须在 (0, 1] 之间")
        self._state: Optional[SE3] = None

    def reset(self, T_init: Optional[SE3] = None):
        self._state = T_init

    def update(self, T_raw: SE3) -> SE3:
        if self._state is None:
            self._state = _to_safe_SE3(T_raw)
            return self._state
        error_pose = _to_safe_SE3(self._state.inv() * T_raw)
        pose_incr = interp_pose_from_identity(
            error_pose, self.alpha_pos, self.alpha_rot)
        self._state = self._state * pose_incr
        return self._state

    @property
    def value(self) -> Optional[SE3]:
        return self._state


# ============================================================
# 循环时序监控 (诊断 ServoJ 实际发送间隔是否稳定)
# ============================================================

class LoopTimingMonitor:
    """
    统计控制循环的实际周期(dt), 用于诊断"是否偏离 ServoJ 要求的 60Hz~1000Hz
    (命令间隔 1~16ms) 区间"以及"周期是否抖动过大"——这两者都会导致机器人
    侧因为按假定的固定命令周期插值, 而产生明显的物理顿挫/抖动。

    用法: 每个循环末尾调用一次 tick(), 每隔 report_every_s 秒打印一次统计
    (最新/平均/最大 dt, 以及超出 [min_dt_ms, max_dt_ms] 区间的次数占比)。
    """

    def __init__(self, min_dt_ms: float = 1.0, max_dt_ms: float = 16.0,
                 report_every_s: float = 2.0):
        self.min_dt_ms = min_dt_ms
        self.max_dt_ms = max_dt_ms
        self.report_every_s = report_every_s
        self._last_t: Optional[float] = None
        self._dts_ms: list = []
        self._last_report_t: Optional[float] = None

    def tick(self, t: Optional[float] = None) -> Optional[str]:
        """
        记录一次循环时刻, 返回统计报告字符串(仅在到达 report_every_s 间隔时返回,
        否则返回 None), 方便上层按需打印。
        """
        t = t if t is not None else time.monotonic()
        report = None
        if self._last_t is not None:
            self._dts_ms.append((t - self._last_t) * 1000.0)
        self._last_t = t

        if self._last_report_t is None:
            self._last_report_t = t
        elif (t - self._last_report_t) >= self.report_every_s and self._dts_ms:
            arr = np.asarray(self._dts_ms)
            out_of_range = np.mean((arr < self.min_dt_ms)
                                   | (arr > self.max_dt_ms)) * 100.0
            report = (f"[循环时序] 最新={arr[-1]:.1f}ms 均值={arr.mean():.1f}ms "
                      f"最大={arr.max():.1f}ms 超出[{self.min_dt_ms:.0f},{self.max_dt_ms:.0f}]ms区间占比={out_of_range:.0f}%")
            self._dts_ms = []
            self._last_report_t = t
        return report


# ===========================================
# 静止检测: 判断 marker 是否已连续静止超过指定时长
# ===========================================

class MarkerStillnessDetector:
    """
    维护 marker 位姿的滑动时间窗口, 判断"标定板是否已连续静止 window_s 秒"。

    判定方法: 计算窗口内位置相对"窗口均值"的 RMS 偏差, 以及姿态相对"窗口中位
    样本"的角度标准差, 若两者都低于阈值、且窗口跨度 >= window_s, 则认为"静止"。

    ⚠️ 这里特意用 RMS/标准差而不是"窗口内最大偏差"作为统计量: 若用最大偏差,
    该统计量会随窗口内样本数增长而系统性变大(极值统计的性质, 大致按
    sqrt(2*ln(N)) 增长) —— 例如 100Hz 采样、3 秒窗口约有 300 个样本, 即使
    marker 真实噪声的标准差只有 1mm, 300 个样本里"最大成对偏差"的期望值也可能
    达到 3~4mm。这意味着"最大偏差"阈值必须比真实噪声水平大好几倍才能不误判,
    调参非常反直觉、且随 --settle-time/--servo-hz 变化阈值还要跟着变。改用
    RMS/标准差后, 这个统计量基本不随样本数变化, 数值直接对应"噪声的真实幅度",
    调参更符合直觉、也更容易和实测噪声水平对齐。

    last_pos_std_m / last_ang_std_deg 记录每次 update() 观测到的实时噪声水平,
    供上层打印诊断信息、依据实测数据设置阈值。
    """

    def __init__(
        self,
        window_s: float = 3.0,
        pos_thresh_m: float = 0.003,
        ang_thresh_deg: float = 2.0,
        min_samples: int = 5,
        span_tolerance: float = 0.95,
    ):
        self.window_s = window_s
        self.pos_thresh_m = pos_thresh_m
        self.ang_thresh_deg = ang_thresh_deg
        self.min_samples = min_samples
        # 固定采样周期下, "窗口内样本跨度"几乎不可能精确等于 window_s(例如
        # dt=14ms 时最大只能凑到 71*14ms=994ms, 永远差一点点到 1000ms),
        # 因此允许 span 达到 window_s 的 span_tolerance 倍即视为"窗口已填满",
        # 否则静止判定会因为这点离散化误差而永远无法触发。
        self.span_tolerance = span_tolerance
        self._buf: Deque[Tuple[float, SE3]] = deque()
        self.last_pos_std_m: float = 0.0
        self.last_ang_std_deg: float = 0.0

    def reset(self):
        self._buf.clear()
        self.last_pos_std_m = 0.0
        self.last_ang_std_deg = 0.0

    def update(self, t: float, base_T_marker: SE3) -> bool:
        """
        喂入一个新的 marker 观测(带时间戳), 返回当前是否判定为"静止"。

        :param t: 时间戳 (s), 建议使用 time.monotonic()
        :param base_T_marker: 当前观测到的 marker 在 base 坐标系下的位姿
        :return: True 表示已连续静止 >= window_s 秒
        """
        self._buf.append((t, base_T_marker))
        # 注意: 这里只裁剪掉"远超窗口(1.5倍余量)"的旧样本, 而不是一超过
        # window_s 就立刻丢弃。原因: 若严格只保留 <= window_s 的样本, 在
        # 固定采样间隔下, 每来一个新样本就会立刻把最老的样本挤出窗口, 稳态下
        # buffer 的实际跨度会永远卡在"略小于 window_s"的地方(每次刚好在
        # 达到 window_s 前一步就被裁掉), span 永远无法达到 window_s,
        # 导致下面的静止判定条件永远为 False —— 这是一个真实存在过的 bug,
        # 和阈值设置无关, 调阈值无法绕开。保留额外余量作为"边界样本", 再在
        # 统计时只筛选窗口内(<= window_s)的样本, 就能让 span 正确达到/超过
        # window_s。
        while self._buf and (t - self._buf[0][0]) > self.window_s * 1.5:
            self._buf.popleft()

        window_samples = [(ts, T)
                          for ts, T in self._buf if (t - ts) <= self.window_s]
        if len(window_samples) < self.min_samples:
            return False

        # 位置: 相对窗口均值的 RMS 偏差 (与样本数基本无关的统计量)
        translations = np.array([T.t for _, T in window_samples])
        mean_t = translations.mean(axis=0)
        pos_std = float(
            np.sqrt(np.mean(np.sum((translations - mean_t) ** 2, axis=1))))

        # 姿态: 相对"窗口中位样本"的角度标准差(用中位样本而非最新样本做基准,
        # 避免基准本身恰好是一个离群噪声点)
        T_ref = window_samples[len(window_samples) // 2][1]
        ang_devs_deg = np.array(
            [np.degrees(pose_error_magnitudes(T_ref, T)[1]) for _, T in window_samples])
        ang_std_deg = float(np.std(ang_devs_deg))

        self.last_pos_std_m = pos_std
        self.last_ang_std_deg = ang_std_deg

        span = window_samples[-1][0] - window_samples[0][0]
        if span < self.window_s * self.span_tolerance:
            return False

        return pos_std <= self.pos_thresh_m and ang_std_deg <= self.ang_thresh_deg

    @property
    def buffer_span_s(self) -> float:
        window_samples = [(ts, T) for ts, T in self._buf if self._buf and (
            self._buf[-1][0] - ts) <= self.window_s]
        if len(window_samples) < 2:
            return 0.0
        return window_samples[-1][0] - window_samples[0][0]


# ============================================================
# 悬停状态机 (带滞回 + 停留时间, 防止远近悬停距离反复横跳)
# ============================================================

class HoverStateMachine:
    """
    TRACKING(远距离跟随) <-> APPROACHING(靠近) 状态切换, 带滞回(hysteresis)
    和最小停留时间(dwell time), 是经典的施密特触发器(Schmitt trigger)思路。

    背景: ArUco solvePnP 对深度(Z轴)方向的位置估计噪声天然比横向大, 即使做了
    低通滤波, 只要噪声幅度接近 MarkerStillnessDetector 的阈值, is_still 就会
    在 True/False 之间反复横跳; 若直接用 is_still 的瞬时值决定悬停距离, 目标
    位姿会在"远"和"近"之间反复跳变, 表现为末端持续小幅度往返移动、迟迟不能
    稳定进入"靠近"状态。

    解决方法: 进入 APPROACHING 用 MarkerStillnessDetector 的(较严)阈值; 退出
    APPROACHING 则需要 marker 相对"进入 APPROACHING 那一刻记录的参考位姿"发生
    明显位移(用更大的 resume 阈值), 且距上次状态切换已超过 min_dwell_s, 两个
    条件都满足才允许切回 TRACKING —— 这样噪声级别的小抖动不会导致状态反复横跳。
    """

    TRACKING = "TRACKING"
    APPROACHING = "APPROACHING"

    def __init__(
        self,
        resume_pos_thresh_m: float,
        resume_ang_thresh_deg: float,
        min_dwell_s: float = 1.0,
    ):
        self.resume_pos_thresh_m = resume_pos_thresh_m
        self.resume_ang_thresh_deg = resume_ang_thresh_deg
        self.min_dwell_s = min_dwell_s
        self.state = self.TRACKING
        self._state_change_t: Optional[float] = None
        self._reference_pose: Optional[SE3] = None

    def reset(self):
        self.state = self.TRACKING
        self._state_change_t = None
        self._reference_pose = None

    def update(self, t: float, is_still: bool, base_T_marker: SE3) -> str:
        """
        :param t: 时间戳(s), 建议 time.monotonic()
        :param is_still: MarkerStillnessDetector.update() 的返回值
        :param base_T_marker: 当前 marker 在 base 坐标系下的位姿(滤波后)
        :return: 更新后的状态, self.TRACKING 或 self.APPROACHING
        """
        if self._state_change_t is None:
            self._state_change_t = t

        if self.state == self.TRACKING:
            if is_still:
                self.state = self.APPROACHING
                self._state_change_t = t
                self._reference_pose = base_T_marker
        else:  # APPROACHING
            dwell_elapsed = (t - self._state_change_t) > self.min_dwell_s
            if dwell_elapsed and self._reference_pose is not None:
                pos_dev, rot_dev = pose_error_magnitudes(
                    self._reference_pose, base_T_marker)
                moved_significantly = (
                    pos_dev > self.resume_pos_thresh_m or
                    np.degrees(rot_dev) > self.resume_ang_thresh_deg
                )
                if moved_significantly:
                    self.state = self.TRACKING
                    self._state_change_t = t
                    self._reference_pose = None

        return self.state
