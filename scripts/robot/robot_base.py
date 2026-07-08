import abc
import numpy as np
from typing import Optional, List, Tuple, Union
from spatialmath import SE3

# 类别名称
JointPos = Union[List[float], np.ndarray]   # 关节位置(弧度)
JointVel = Union[List[float], np.ndarray]   # 关节速度(弧度/秒)
JointTorq = Union[List[float], np.ndarray]  # 关节力矩(Nm)
JointLimit = Tuple[float, float]            # 关节限制(最小值, 最大值)
Pose = SE3                                  # 位姿(空间坐标系)
JointTrajectory = List[JointPos]
PoseTrajectory = List[Pose]

class RobotBase(abc.ABC):
    """
    机器人基类, 定义工业机器人/协作机器人的核心接口
    """

    def __init__(self, model_name: str = "unknown_robot", degree_of_freedom: int = 6):
        """
        初始化机器人基类
        :param model_name: 机器人型号名称(如: FR5, FR3C, JAKA 系列, UR 系列, 等)
        :param degree_of_freedom: 机器人自由度(关节数)
        """
        self._model_name = model_name # 机器人型号名称
        self._degree_of_freedom = degree_of_freedom # 机器人自由度
        self._q:JointPos = None                     # 当前关节位置(弧度)
        self._dq:JointVel = None                    # 当前关节速度(弧度/秒)
        self._pose:Pose = None                      # 当前位姿(空间坐标系)
        self._joint_limits:List[JointLimit] = None  # 关节限制(最小值, 最大值)

    
    @abc.abstractmethod
    def forward_kinematics(self, joint_pos: JointPos) -> Pose:
        """
        正运动学计算, 根据关节位置计算末端位姿
        :param joint_pos: 关节位置(弧度)
        :return: 末端位姿(空间坐标系)
        """
        pass


    @abc.abstractmethod
    def inverse_kinematics(self, target_pose: Pose, 
                           reference_joint_pose: Optional[JointPos] = None) -> Optional[JointPos]:
        """
        逆运动学计算, 根据目标位姿计算关节位置
        :param target_pose: 目标位姿(空间坐标系)
        :param reference_joint_pose: 参考关节位置(弧度)
        :return: 关节位置(弧度)
        """
        pass

    
    @abc.abstractmethod
    def move_joint_space(self, 
                         current_joint_pos: JointPos,
                         target_joint_pos: JointPos,
                         velocity: Optional[float] = None,
                         acceleration: Optional[float] = None) -> Optional[JointTrajectory]:
        """
        关节空间运动: 控制关节到达目标关节位置
        :param current_joint_pos: 当前关节位置(弧度)
        :param target_joint_pos: 目标关节位置(弧度)
        :param velocity: 关节运动速度(弧度/秒, 可选, 默认使用机器人最大速度)
        :param acceleration: 关节运动加速度(弧度/秒^2, 可选, 默认使用机器人最大加速度)
        """
        pass


    @abc.abstractmethod
    def move_to_pose(self,
                     target_pose: Pose,
                     reference_joint_pose: Optional[JointPos] = None) -> Optional[PoseTrajectory]:
        """
        任务空间运动: 控制末端到达目标位姿
        :param target_pose: 目标位姿(空间坐标系)
        :param reference_joint_pose: 参考关节位置(弧度, 可选, 默认使用当前关节位置)
        """
        pass


    def get_dof(self) -> int:
        """
        获取机器人自由度(关节数)
        :return: 机器人自由度(关节数)
        """
        return self._degree_of_freedom
    

    def get_model_name(self) -> str:
        """
        获取机器人型号名称
        :return: 机器人型号名称
        """
        return self._model_name