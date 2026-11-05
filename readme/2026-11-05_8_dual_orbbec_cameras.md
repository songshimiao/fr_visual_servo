# Gemini 305 / 335 双相机配置与 Web UI（2026-11-05）

## 固定安装方式

| 型号 | 当前序列号 | 安装方式 | 手眼矩阵含义 |
|---|---|---|---|
| Gemini 305 | CV27864000EE | eye_in_hand | flange_T_camera，将相机点变换至 tool0 法兰，单位米 |
| Gemini 335 | CP08863000Z7 | eye_to_hand | base_T_camera，将相机点变换至机器人基座，单位米 |

默认 `--camera orbbec`，主相机为 305，另一台为 335。按型号筛选 USB 设备，
不会依赖枚举顺序。多台同型号设备时必须显式指定序列号：
`--camera-serial` 对应主相机，`--secondary-camera-serial` 对应另一台。
`--camera-model gemini335` 可交换主副显示顺序，但不改变安装方式。
PBVS 和悬停入口仍要求主视觉为 eye_in_hand；335 只作为外部监控相机，不参与控制融合。

## 运行

在 Jetson 的 `/home/jetson/fairino_ws/fr_visual_servo` 目录执行，同一时间只运行一个相机入口。

```bash
# 双画面监控，不连接机器人
python3 scripts/camera_preview.py

# 双画面 ArUco 检测（305 检测叠加、335 原始监控）
python3 scripts/algorithm/aruco_detector.py --marker-length 0.05

# 两台相机独立采集内参，默认各 20 张
python3 scripts/algorithm/camera_calibration.py intrinsic --num-images 20

# 两台相机独立采集手眼位姿，默认各 15 组，只读取关节角，不自动移动机器人
python3 scripts/algorithm/camera_calibration.py extrinsic --num-poses 15

# 伺服作业，默认 dry-run；沿用现有机械臂连接和执行确认
python3 scripts/main_pbvs.py

# 单相机兼容入口
python3 scripts/camera_preview.py --single-camera --camera-model gemini335
python3 scripts/camera_preview.py --camera realsense
```

网页地址为 `http://192.168.66.80:8080/`，宽屏并排显示，窄屏上下排列。
停止当前预览后再启动标定或伺服入口；可点击网页“结束当前程序/采集”结束纯相机预览。

## 标定保存行为

- 内参/手眼标定页面的每个画面下均有独立“保存当前样本”按钮，显示本路保存数量和反馈。
- `s` 仅保存主相机样本；`q` 和结束按钮通知两路结束。按钮提交采样请求，采集线程处理当时的有效图像。
- 内参样本必须有足够的 ChArUco 角点；手眼样本必须能估计标定板位姿。无效请求显示原因，不增加计数。
- 两路相机没有硬件同步，不能将两次独立保存视为同一时刻；采样时应让机械臂和标定板静止。
- 305 手眼采集时标定板相对基座固定；335 手眼采集时标定板相对法兰固定。必须按对应安装约定分别采样；可以使用两块固定方式不同的板，或分阶段采集。
- 一路达到目标后独立计算并保存，继续显示实时画面，等待另一路完成。
- 内参少于 4 张、手眼少于 3 组时不会写入该路的标定文件。提前结束时，满足最低数量的该路仍按原流程求解并保存。
- 伺服、悬停、ArUco 检测及原始预览页面不显示保存按钮，保存 API 也拒绝采样。
- 网页结束沿用各入口原有退出逻辑，机器人作业中不是急停。

## 文件隔离

| 数据 | Gemini 305 | Gemini 335 |
|---|---|---|
| 内参 | `resources/calibration/camera_intrinsics_orbbec_gemini305.json` | `resources/calibration/camera_intrinsics_orbbec_gemini335.json` |
| 手眼 | `resources/calibration/hand_eye_calibration_orbbec_gemini305.json` | `resources/calibration/hand_eye_calibration_orbbec_gemini335.json` |
| 内参原图 | `resources/intrinsics_img/gemini305/<session>/` | `resources/intrinsics_img/gemini335/<session>/` |

双相机模式下 `--out` / `--intrinsics` 指定主相机文件，另一台使用表中默认文件。
`--save-images-dir` 修改原图根目录，其下继续按型号和采集会话隔离。
文件加载会检查型号、安装方式、序列号和图像尺寸；不会自动缩放标定参数。

当前 305 和新导出的 335 内参均来自各自 SDK 出厂参数（`source=factory_sdk`），
不代表已完成 ChArUco 精度验证。两台相机的手眼标定结果仍需现场采样生成。
保留原 RealSense 文件，未用它们替代 Gemini 的标定。

## 实现和验证

共用 `WebPreview.from_args()` 为监控入口增加第二路独立采集线程和编码缓冲区，
因此网页客户端或第二路 JPEG 编码不在机器人控制循环中执行。
标定流程使用两个采集线程，各自处理按钮事件和样本，机械臂关节读取串行化。
退出时释放相机、编码线程和 HTTP 端口；任一路相机异常会通知全局停止。

在 Jetson 的系统 Python 3.10 下执行：

```bash
python3 -m unittest discover -s tests -v
```

27 项测试通过，覆盖设备型号筛选、标定隔离、两路真实 HTTP/JPEG、独立保存、
监控禁止采样、相机启动失败释放资源及原有 dry-run/取消/检测失败不运动逻辑。
已在实际 305 和 335 上同时打开 `640×480@30` 彩色流，并经 HTTP 解码两路
`480×640×3` 图像；实际标定页面的 335 保存请求未改变 305 计数，
无足够角点时显示未保存原因，样本不足未覆盖标定文件。

未执行机械臂运动、未验证新的手眼精度和实机闭环伺服性能。
