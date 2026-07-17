"""
Fairino (FR) 系列协作机器人 MDH (Modified/Craig Denavit-Hartenberg) 参数表
数据来源: https://fairino-doc-zhs.readthedocs.io/latest/CobotsManual/robot_brief_introduction.html
        2.4 机器人 DH 参数
约定: 
    - 手册表格列顺序为: theta[rad], a[mm], d[mm], alpha[rad]
    - theta 列在手册中全部为 0, 即零位关节偏置 (offset) 为 0
    - a, d 原始单位为 mm, 这里统一换算成 m, 供 roboticstoolbox / spatialmath 使用
    - 对应 roboticstoolbox 的 rtb.RevoluteMDH(a=a, d=d, alpha=alpha, offset=theta)
    - FR 系列机器人在做姿态或坐标系变换时，齐次变换矩阵计算的角度旋转顺序为浮动坐标系的 Z-Y-X 顺序, 即 R = Rz * Ry * Rx
"""

import numpy as np
from typing import Dict, List, NamedTuple


class MDHParam(NamedTuple):
    """
    单个关节的 Modified DH 参数
    """
    theta_offset: float  # 关节零位偏置 (rad)
    a: float             # 连杆长度 (m)
    d: float             # 连杆偏距 (m)
    alpha: float         # 连杆扭转角 (rad)


PI = np.pi
HALF_PI = np.pi / 2.0


def _mm(v: float) -> float:
    """
    将毫米(mm)单位转换为米(m)单位
    :param v: 毫米(mm)值
    :return: 米(m)值
    """
    return v / 1000.0


# ----------------------------------------------------
# 各型号 MDH 参数表: theta(=0), a[mm], d[mm], alpha[rad]
# 数值取自官方手册 Table 2.4-1 ~ Table 2.4-12
# ----------------------------------------------------
FAIRINO_MDH_PARAMS: Dict[str, List[MDHParam]] = {
    # Table 2.4-1
    "FR3": [
        MDHParam(0.0, _mm(0.0),    _mm(140.0),  HALF_PI),
        MDHParam(0.0, _mm(-280.0), _mm(0.0),        0.0),
        MDHParam(0.0, _mm(-240.0), _mm(0.0),        0.0),
        MDHParam(0.0, _mm(0.0),    _mm(102.0),  HALF_PI),
        MDHParam(0.0, _mm(0.0),    _mm(102.0), -HALF_PI),
        MDHParam(0.0, _mm(0.0),    _mm(100.0),      0.0),
    ],
    # Table 2.4-2
    "FR3-WMS": [
        MDHParam(0.0, _mm(140.0), _mm(0.0),  HALF_PI),
        MDHParam(0.0, _mm(0.0),   _mm(-280.0),   0.0),
        MDHParam(0.0, _mm(0.0),   _mm(-240.0),   0.0),
        MDHParam(0.0, _mm(102.0), _mm(0.0),  HALF_PI),
        MDHParam(0.0, _mm(102.0), _mm(0.0), -HALF_PI),
        MDHParam(0.0, _mm(100.0), _mm(0.0),      0.0),
    ],
    # Table 2.4-3
    "FR3-WML": [
        MDHParam(0.0, _mm(140.0), _mm(0.0),  HALF_PI),
        MDHParam(0.0, _mm(0.0),   _mm(-425.0),   0.0),
        MDHParam(0.0, _mm(0.0),   _mm(-395.0),   0.0),
        MDHParam(0.0, _mm(102.0), _mm(0.0),  HALF_PI),
        MDHParam(0.0, _mm(102.0), _mm(0.0), -HALF_PI),
        MDHParam(0.0, _mm(100.0), _mm(0.0),      0.0),
    ],
    # Table 2.4-4
    "FR3-C": [
        MDHParam(0.0, _mm(140.0), _mm(0.0),  HALF_PI),
        MDHParam(0.0, _mm(0.0),   _mm(-280.0),   0.0),
        MDHParam(0.0, _mm(0.0),   _mm(-240.0),   0.0),
        MDHParam(0.0, _mm(102.0), _mm(0.0),  HALF_PI),
        MDHParam(0.0, _mm(102.0), _mm(0.0), -HALF_PI),
        MDHParam(0.0, _mm(100.0), _mm(0.0),      0.0),
    ],
    # Table 2.4-5
    "FR5": [
        MDHParam(0.0, _mm(0.0),    _mm(152.0),      0.0),
        MDHParam(0.0, _mm(0.0),    _mm(0.0),    HALF_PI),
        MDHParam(0.0, _mm(-425.0), _mm(0.0),        0.0),
        MDHParam(0.0, _mm(-395.0), _mm(102.0),      0.0),
        MDHParam(0.0, _mm(0.0),    _mm(102.0),  HALF_PI),
        MDHParam(0.0, _mm(0.0),    _mm(100.0), -HALF_PI),
    ],
    # Table 2.4-6
    "FR5-C": [
        MDHParam(0.0, _mm(0.0),    _mm(140.0),  HALF_PI),
        MDHParam(0.0, _mm(-280.0), _mm(0.0),        0.0),
        MDHParam(0.0, _mm(-240.0), _mm(0.0),        0.0),
        MDHParam(0.0, _mm(0.0),    _mm(102.0),  HALF_PI),
        MDHParam(0.0, _mm(0.0),    _mm(102.0), -HALF_PI),
        MDHParam(0.0, _mm(0.0),    _mm(100.0),      0.0),
    ],
    # Table 2.4-7
    "FR5-WML": [
        MDHParam(0.0, _mm(180.0), _mm(0.0),  HALF_PI),
        MDHParam(0.0, _mm(0.0),   _mm(-970.0),   0.0),
        MDHParam(0.0, _mm(0.0),   _mm(-816.0),   0.0),
        MDHParam(0.0, _mm(159.0), _mm(0.0),  HALF_PI),
        MDHParam(0.0, _mm(114.0), _mm(0.0), -HALF_PI),
        MDHParam(0.0, _mm(160.0), _mm(0.0),      0.0),
    ],
    # Table 2.4-8
    "FR10": [
        MDHParam(0.0, _mm(0.0),    _mm(180.0),  HALF_PI),
        MDHParam(0.0, _mm(-700.0), _mm(0.0),        0.0),
        MDHParam(0.0, _mm(-586.0), _mm(0.0),        0.0),
        MDHParam(0.0, _mm(0.0),    _mm(159.0),  HALF_PI),
        MDHParam(0.0, _mm(0.0),    _mm(114.0), -HALF_PI),
        MDHParam(0.0, _mm(0.0),    _mm(106.0),      0.0),
    ],
    # Table 2.4-9
    "FR16": [
        MDHParam(0.0, _mm(0.0),    _mm(180.0),  HALF_PI),
        MDHParam(0.0, _mm(-520.0), _mm(0.0),        0.0),
        MDHParam(0.0, _mm(-400.0), _mm(0.0),        0.0),
        MDHParam(0.0, _mm(0.0),    _mm(159.0),  HALF_PI),
        MDHParam(0.0, _mm(0.0),    _mm(114.0), -HALF_PI),
        MDHParam(0.0, _mm(0.0),    _mm(106.0),      0.0),
    ],
    # Table 2.4-10
    "FR20": [
        MDHParam(0.0, _mm(0.0),     _mm(215.0),  HALF_PI),
        MDHParam(0.0, _mm(-1000.0), _mm(0.0),        0.0),
        MDHParam(0.0, _mm(-716.0),  _mm(0.0),        0.0),
        MDHParam(0.0, _mm(0.0),     _mm(166.0),  HALF_PI),
        MDHParam(0.0, _mm(0.0),     _mm(138.0), -HALF_PI),
        MDHParam(0.0, _mm(0.0),     _mm(120.0),      0.0),
    ],
    # Table 2.4-11
    "FR30": [
        MDHParam(0.0, _mm(0.0),    _mm(215.0),  HALF_PI),
        MDHParam(0.0, _mm(-700.0), _mm(0.0),        0.0),
        MDHParam(0.0, _mm(-536.0), _mm(0.0),        0.0),
        MDHParam(0.0, _mm(0.0),    _mm(166.0),  HALF_PI),
        MDHParam(0.0, _mm(0.0),    _mm(138.0), -HALF_PI),
        MDHParam(0.0, _mm(0.0),    _mm(120.0),      0.0),
    ],
    # Table 2.4-12
    "FR30L": [
        MDHParam(0.0, _mm(0.0),     _mm(215.0),  HALF_PI),
        MDHParam(0.0, _mm(-1000.0), _mm(0.0),        0.0),
        MDHParam(0.0, _mm(-716.0),  _mm(0.0),        0.0),
        MDHParam(0.0, _mm(0.0),     _mm(166.0),  HALF_PI),
        MDHParam(0.0, _mm(0.0),     _mm(138.0), -HALF_PI),
        MDHParam(0.0, _mm(0.0),     _mm(120.0),      0.0),
    ],
}


# -------
# 关节限位
# -------
DEFAULT_QLIM_DEG = [
    (-175.0, 175.0), # J1
    (-265.0,  85.0), # J2
    (-155.0, 155.0), # J3
    (-260.0,  85.0), # J4
    (-175.0, 175.0), # J5
    (-175.0, 175.0), # J6
]

DEFAULT_QLIM_RAD = [(np.deg2rad(qmin), np.deg2rad(qmax)) for qmin, qmax in DEFAULT_QLIM_DEG]

FR3_C_QLIM_DEG = [
    (-175.0, 175.0), # J1
    (-265.0,  85.0), # J2
    (-150.0, 150.0), # J3
    (-265.0,  85.0), # J4
    (   0.0, 355.0), # J5
    (-175.0, 175.0), # J6
]

FR3_C_QLIM_RAD = [(np.deg2rad(qmin), np.deg2rad(qmax)) for qmin, qmax in FR3_C_QLIM_DEG]


def get_supported_models() -> List[str]:
    """
    获取当前已收录 MDH 参数的 Fairino 机器人型号列表
    :return: 支持的 Fairino 机器人型号列表
    """
    return list(FAIRINO_MDH_PARAMS.keys())


def get_mdh_params(model_name: str) -> List[MDHParam]:
    """
    按型号名称获取 MDH 参数
    :param model_name: Fairino 机器人型号名称, 如: FR3, FR5, FR10 等 (大小写不敏感)
    :return: MDH 参数列表, 按关节顺序排列
    :raises KeyError: 型号不存在时抛出, 并给出当前支持的型号列表
    """
    key = model_name.strip().upper()
    # 支持 "FR5", "fr5", "Fr5-C" 等大小写混合的型号名称
    normalized = {k.upper(): k for k in FAIRINO_MDH_PARAMS}
    if key not in normalized:
        raise KeyError(
            f" Unknown Fairino robot: '{model_name}'."
            f" Supported models: {get_supported_models()}"
        )
    return FAIRINO_MDH_PARAMS[normalized[key]]

if __name__ == "__main__":
    # 测试: 打印所有支持的型号及其 MDH 参数
    for model in get_supported_models():
        print(f"Model: {model}")
        params = get_mdh_params(model)
        for i, p in enumerate(params):
            print(f"  Joint {i+1}: theta_offset={p.theta_offset}, a={p.a}, d={p.d}, alpha={p.alpha}")
        print()