# FR5 视觉伺服控制系统

这是一个基于位置的视觉伺服（PBVS）系统，用于控制 Fairino FR5 机器人实时跟随 ArUco 标记板进行视觉引导操作。

## 系统概述

本项目实现了基于 RealSense D435 相机的视觉伺服控制系统，能够：
- 实时检测 ArUco 标记板
- 通过视觉伺服算法控制机器人末端精确跟随标记板
- 支持"远距离跟随"和"近距离靠近"两种悬停模式
- 具备静止检测功能，当标记板静止时自动切换到靠近模式
- 提供完整的安全机制和参数调节选项
- 新增"look-then-move"盲逼近功能，解决近距离视觉伺服问题
- **新增PnP多解消除功能，解决相机正对时的视觉伺服抖动问题** ([详情](readme/2026-07-09_1_pnp_multisolution_elimination.md))
- **新增机械臂运动抖动解决方案** ([详情](readme/2026-07-10_2_mechanical_jitter_solution.md))
- **新增项目近期改动记录** ([详情](readme/2026-07-17_3_project_changes.md))

## 项目结构

```
fr_visual_servo/
├── scripts/
│   ├── main_pbvs.py              # 主程序入口
│   ├── algorithm/
│   │   ├── aruco_detector.py     # ArUco 标记检测模块
│   │   └── Pbvs.py               # PBVS 控制算法核心
│   └── robot/
│       ├── fairino_robot.py      # Fairino 机器人运动学实现
│       ├── robot_base.py         # 机器人基类定义
│       └── fairino/              # Fairino 机器人SDK相关文件
├── resources/
│   ├── calibration/              # 标定数据文件
│   └── intrinsics_img/           # 内参标定图片
└── README.md                     # 本文档
```

## 功能特点

### 1. 双线程架构设计
- **相机线程**：负责实时图像采集和 ArUco 标记检测（约30fps）
- **控制线程**：负责视觉伺服控制和机器人运动（可配置频率，默认100Hz）

### 2. 高级控制算法
- 基于位置的视觉伺服（PBVS）控制律
- 位姿误差计算和插值控制
- 工作空间限制保护
- 低通滤波平滑处理
- 状态机管理（远近悬停切换）
- 性能优化：减少SVD计算开销，提升控制循环效率

### 3. 特殊功能
- **盲逼近模式**：新增 `--blind-final-approach` 参数，实现"look-then-move"策略，解决近距离时marker超出视野的问题
- **PnP多解消除**：解决相机正对marker时的解跳变问题，提升视觉伺服稳定性
- **运动平滑优化**：解决机械臂启动和状态切换时的抖动问题
- **标定精度提升**：基于更新的标定数据，提高系统定位精度

## 使用方法

### 1. 运行主程序

```bash
# 基本使用
python scripts/main_pbvs.py

# 实际执行控制（需要确保安全）
python scripts/main_pbvs.py --execute

# 指定参数运行
python scripts/main_pbvs.py --intrinsics resources/calibration/camera_intrinsics.json \
                           --hand-eye resources/calibration/hand_eye_calibration.json \
                           --execute
```

### 2. 主要参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--intrinsics` | `resources/calibration/camera_intrinsics.json` | 相机内参文件路径 |
| `--hand-eye` | `resources/calibration/hand_eye_calibration.json` | 手眼标定文件路径 |
| `--execute` | - | 实际下发运动指令（默认为dry-run模式） |
| `--servo-hz` | 100.0 | 控制频率（Hz） |
| `--hover-far` | 0.15 | 远距离悬停高度（米） |
| `--hover-near` | 0.05 | 近距离悬停高度（米） |
| `--blind-final-approach` | - | 启用"look-then-move"盲逼近模式，解决近距离视觉伺服问题 |

## 系统依赖

### Python 依赖
```bash
pip install opencv-contrib-python pyrealsense2 numpy roboticstoolbox spatialmath-python
```

### 硬件需求
- Fairino FR5 机器人
- Intel RealSense D435 相机
- 标定板（用于相机和手眼标定）

## 标定说明

系统使用以下标定文件：
1. `camera_intrinsics.json` - 相机内参
2. `hand_eye_calibration.json` - 手眼标定结果

需要使用配套的标定工具生成这些文件。

## 安全注意事项

⚠️ **重要提醒**：
1. 默认为 dry-run 模式，仅打印计算结果不下发指令
2. 使用 `--execute` 参数前，必须确认机器人周围安全
3. 请确保急停开关处于可触及范围
4. 建议先用 `--diagnose-timing` 参数检查控制周期稳定性

## 控制逻辑

1. **远距离跟随模式**：机器人在标记板上方一定距离跟随
2. **静止检测**：当标记板连续静止超过设定时间后，自动切换到近处跟随
3. **重新运动检测**：当标记板重新运动时，返回远距离跟随模式
4. **盲逼近模式**：当启用 `--blind-final-approach` 参数时，在标记板静止时冻结目标位姿，退出视觉反馈，纯运动学伺服到目标位置

## 技术细节

### 核心算法组件
- **ArUco 检测**：使用 OpenCV 实现
- **视觉伺服**：基于 SO3 插值的 PBVS 控制
- **运动学**：使用 Robotics Toolbox for Python 实现正/逆运动学
- **状态管理**：包含静止检测和悬停状态机

### 技术优化
- PnP 多解消除：解决相机正对时的抖动问题
- 时序优化：相机检测与控制分离保证实时性
- 误差处理：包含 NaN/Inf 检测和处理
- 参数限幅：防止过度运动导致的不稳定
- 性能优化：使用快速旋转角度计算替换耗时的 SVD 正交化
- **运动抖动优化**：解决机械臂启动和状态切换时的剧烈抖动问题
- **标定精度提升**：基于更新的标定数据，提高系统定位精度

## 开发与调试

### 调试选项
```bash
# 显示详细时序信息
python scripts/main_pbvs.py --diagnose-timing

# 降低控制频率进行测试
python scripts/main_pbvs.py --servo-hz 50

# 启用盲逼近模式
python scripts/main_pbvs.py --blind-final-approach

# 查看帮助信息
python scripts/main_pbvs.py --help
```

### 常见问题
1. **相机无法连接**：检查USB连接和驱动
2. **机器人连接失败**：确认IP地址和网络连通性
3. **控制不稳定**：调整增益参数或提高控制频率
4. **标记识别困难**：检查照明条件和标记对比度

## 版本信息

- 支持 Fairino FR5 机器人
- 基于 Python 3.8+
- 依赖 OpenCV 4.x 和 Robotics Toolbox
- 当前版本包含性能优化、盲逼近功能、运动抖动解决方案和最新的标定数据