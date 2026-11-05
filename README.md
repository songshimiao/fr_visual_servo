# FR5 视觉伺服控制系统

这是一个基于位置的视觉伺服（PBVS）系统，用于控制 Fairino FR5 机器人实时跟随 ArUco 标记板进行视觉引导操作。

## 系统概述

本项目支持 Intel RealSense D435 和 Orbbec Gemini 305，相机入口启动时可在终端选择，能够：
- 实时检测 ArUco 标记板
- 通过视觉伺服算法控制机器人末端精确跟随标记板
- 支持"远距离跟随"和"近距离靠近"两种悬停模式
- 具备静止检测功能，当标记板静止时自动切换到靠近模式
- 提供完整的安全机制和参数调节选项
- 新增"look-then-move"盲逼近功能，解决近距离视觉伺服问题
- **新增PnP多解消除功能，解决相机正对时的视觉伺服抖动问题** ([详情](readme/2026-07-09_1_pnp_multisolution_elimination.md))
- **新增机械臂运动抖动解决方案** ([详情](readme/2026-07-10_2_mechanical_jitter_solution.md))
- **新增项目近期改动记录** ([详情](readme/2026-07-17_3_project_changes.md))
- **新增关节限位检查功能** ([详情](readme/2026-07-17_4_joint_limit_check.md))
- **支持 Jetson 无显示器开发和网页视频监控** ([使用与改动记录](readme/2026-10-21_5_headless_web_monitoring.md))
- **支持 Gemini 305 / RealSense 切换** ([环境配置、命令和实测记录](readme/2026-10-21_6_orbbec_gemini305.md))

## 项目结构

```
fr_visual_servo/
├── scripts/
│   ├── main_pbvs.py              # 主程序入口
│   ├── camera_preview.py         # 双相机原始视频、自检和出厂内参导出
│   ├── setup_orbbec_jetson.sh     # Jetson Orbbec SDK v2 + USB 权限配置
│   ├── algorithm/
│   │   ├── camera_stream.py      # RealSense / Orbbec 共用采集接口和终端选择
│   │   ├── aruco_detector.py     # ArUco 标记检测模块
│   │   ├── web_preview.py        # 无显示器 MJPEG 网页预览与标定交互
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
- **关节限位检查**：主动检查关节位置是否超出安全范围，避免机械臂超限故障

## 使用方法

### Jetson 无显示器远程监控

相机入口默认启动网页视频流，不再创建 OpenCV 本地窗口。在 Jetson 项目根目录运行：

```bash
# 终端选择 1=RealSense、2=Gemini 305；不连接机械臂
# marker-length 按实际边长设置（米）
python3 scripts/algorithm/aruco_detector.py \
  --marker-length 0.05

# PBVS 主程序（默认 dry-run，需要机械臂在线）
python3 scripts/main_pbvs.py
```

电脑浏览器访问 **http://192.168.66.80:8080/**。支持 `--web-port`、`--web-fps`、
`--web-quality`、`--web-host` 和 `--no-web`。标定时在网页点击“采集一帧”或按 s；
网页 q/结束按钮触发原有退出流程，执行模式下可能包含回位动作，并非急停。
机器人执行确认仍在 SSH 终端进行。服务用于可信局域网，也可通过 SSH 转发访问。

完整命令、标定操作、退出行为及离线测试见
[无显示器网页监控记录](readme/2026-10-21_5_headless_web_monitoring.md)。

### 相机选择与 Gemini 305 环境

在 Jetson 项目根目录执行。当前 Jetson 已安装 `pyorbbecsdk2==2.1.2`（底层 SDK 2.9.3）
和 Orbbec USB udev 规则，原有 RealSense、OpenCV、NumPy 环境保留。
重新配置或迁移时运行：

```bash
bash scripts/setup_orbbec_jetson.sh

# 不带 --camera 会在终端选择相机；网页地址仍为 http://192.168.66.80:8080/
python3 scripts/camera_preview.py
python3 scripts/algorithm/aruco_detector.py --marker-length 0.05

# 可直接指定；非交互运行时必须指定 --camera
python3 scripts/algorithm/aruco_detector.py --camera orbbec --marker-length 0.05
python3 scripts/algorithm/aruco_detector.py --camera realsense --marker-length 0.05

# Gemini 实测内参标定：网页采集，自动输出到 Gemini 专用文件
python3 scripts/algorithm/camera_calibration.py intrinsic --camera orbbec \
  --num-images 20 --save-images-dir resources/intrinsics_img/orbbec_gemini305
```

Gemini 默认内参为 `resources/calibration/camera_intrinsics_orbbec_gemini305.json`，
已从当前相机读取出厂参数并标注 `source=factory_sdk`；可先用于检测，尚未进行 ChArUco 精度验证。
Gemini 手眼文件默认为 `resources/calibration/hand_eye_calibration_orbbec_gemini305.json`，
**需要针对新相机的安装重新标定，当前尚未生成**。RealSense 继续使用原来的文件名。
PBVS 和悬停入口同样支持终端选择及 `--camera`；机械臂联动未做实机验证。
Jetson 的 `spatialmath/roboticstoolbox` 机器人依赖现已安装，版本约束见
`requirements-jetson.txt`；实际运动仍需在终端确认现场安全。

彩色流默认为 `640×480@30`，支持 `--camera-width / --camera-height / --camera-fps`，
多台同型号相机可用 `--camera-serial`。更换分辨率需要对应内参；不支持的 Orbbec 流配置
会列出可用配置，不自动缩放图像。详细安装过程、出厂内参导出、重新标定和实测结果见
[Gemini 305 接入记录](readme/2026-10-21_6_orbbec_gemini305.md)。

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
| `--camera` | 终端选择 | `realsense` 或 `orbbec`；非交互必须指定 |
| `--intrinsics` | 按相机选择 | RealSense 使用原文件；Gemini 使用 `_orbbec_gemini305.json` 后缀 |
| `--hand-eye` | 按相机选择 | PBVS/悬停按相机选择；独立 ArUco 默认不加载手眼 |
| `--execute` | - | 实际下发运动指令（默认为dry-run模式） |
| `--servo-hz` | 100.0 | 控制频率（Hz） |
| `--hover-far` | 0.15 | 远距离悬停高度（米） |
| `--hover-near` | 0.05 | 近距离悬停高度（米） |
| `--blind-final-approach` | - | 启用"look-then-move"盲逼近模式，解决近距离视觉伺服问题 |
| `--joint-limit-margin-deg` | 3.0 | 关节限位安全裕度(度)，提前预警的缓冲区大小 |

## 系统依赖

### Python 依赖（Jetson Python 3.10）
```bash
python3 -m pip install --user -r requirements-jetson.txt
```

现有 Jetson 已修复 NumPy ABI 冲突，直接使用系统 `python3` 即可。NumPy 固定为
`1.26.4`，`pgraph-python` 固定为 `0.6.5`：后者的 `1.0.0` 要求 NumPy >=2，
会与当前 OpenCV 二进制及 Orbbec SDK 冲突。不要单独升级这两个包；安装其他包时可加
`-c requirements-jetson.txt` 保持已验证的版本。机器人库的 pip 包名是
`roboticstoolbox-python`，导入名是 `roboticstoolbox`。
修复原因、运行命令和实测结果见 [Jetson PBVS 依赖修复记录](readme/2026-10-22_7_jetson_pbvs_dependencies.md)。

已有 Jetson 环境请勿重复升级 OpenCV/NumPy；Gemini 新增运行时使用
`bash scripts/setup_orbbec_jetson.sh` 安装，SDK 在选择对应相机后才导入。

### 硬件需求
- Fairino FR5 机器人
- Intel RealSense D435 或 Orbbec Gemini 305 相机（USB 3）
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
- **关节限位保护**：主动检查关节位置安全范围，避免机械臂超限故障

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
- 当前版本包含性能优化、盲逼近功能、运动抖动解决方案、最新的标定数据和关节限位保护功能
