# Jetson 无显示器开发：RealSense 网页视频监控

记录日期：2026-10-21。

## 背景和结果

Jetson 通过 SSH 远程开发时没有显示器、桌面会话或 DISPLAY。原来的
`cv2.imshow / waitKey / destroyAllWindows` 已从工程的相机业务脚本中移除，
改为 HTTP MJPEG 视频流。电脑浏览器可以查看 RealSense 彩色图像、ArUco 检测
叠加和 PBVS 状态，也可以提交标定采样、结束请求。

不需要 Flask、FastAPI、FFmpeg、浏览器插件或 X11 转发。使用 Python 标准库
HTTP 服务和现有 OpenCV JPEG 编码，仍需原有 numpy、RealSense、ArUco 等依赖。
不要为了本次改动直接替换 Jetson 上已有的 OpenCV/RealSense 安装。

## 修改文件

| 文件 | 修改内容 |
| --- | --- |
| `scripts/algorithm/web_preview.py` | 共用 MJPEG 服务、网页、异步编码、采样/结束事件、无硬件自检入口 |
| `scripts/main_pbvs.py` | 网页替代视觉线程的本地窗口；相机/服务失败通知主循环退出 |
| `scripts/algorithm/aruco_detector.py` | 独立检测入口发布网页视频，保留检测与坐标轴叠加 |
| `scripts/algorithm/camera_calibration.py` | 内参、手眼标定通过网页按钮或网页 s/q 键交互 |
| `scripts/visual_servo_hover_test.py` | 稳定检测阶段发布叠加图像，之后后台继续发布相机画面；修正清理阶段的执行条件 |
| `tests/test_web_preview.py` | 无相机、无机器人条件下的 HTTP/JPEG 和相机调用点回归测试 |
| `README.md` | 增加无显示器启动说明和本记录链接 |

本次未调整 PBVS 控制律、FK/IK、标定数据和伺服频率。原先未提交的主程序收敛阈值
`0.0005 m / 0.1 deg` 保留。

## 在 Jetson 启动

以下命令均在 Jetson 的项目根目录执行：

```bash
cd /home/jetson/fairino_ws/fr_visual_servo

# 实时视觉伺服主程序，默认 dry-run；需要相机和机器人在线
python3 scripts/main_pbvs.py

# 仅检查相机、标记识别和网页视频，不连接机械臂
python3 scripts/algorithm/aruco_detector.py \
  --intrinsics resources/calibration/camera_intrinsics.json \
  --marker-length 0.05
```

电脑浏览器打开 **http://192.168.66.80:8080/**。如果 Jetson IP 变化，请替换地址。
`--marker-length` 单位为米，必须使用实际标记边长；独立检测示例采用 5 cm。
图像来自彩色流，不增加深度流采集。

主程序实际运动仍需原有 `--execute` 参数和 **SSH 终端中的 yes 确认**。
网页不能使能机器人或开启执行模式。不要同时启动多个脚本占用同一台 RealSense；
不同进程使用不同相机时也应配置不同网页端口。

## 标定和悬停验证

```bash
# 内参标定：在网页点击“采集一帧”，或聚焦网页按 s
python3 scripts/algorithm/camera_calibration.py intrinsic \
  --num-images 20 --save-images-dir resources/intrinsics_img

# 手眼标定：每次调整机械臂姿态后在网页采集
python3 scripts/algorithm/camera_calibration.py extrinsic \
  --mode eye_in_hand --num-poses 15

# 悬停定位验证，默认只计算、不执行运动
python3 scripts/visual_servo_hover_test.py --mode camera
```

网页采样按钮仅在标定入口显示，采集的是**请求被采集循环处理时的当前帧**，
不是浏览器正在显示的那张压缩图像。点击后以 SSH 终端的“已采集”计数为准；
连续快速点击会合并成一个待处理请求。标定板未检出时不会保存有效标定样本。

标定的 q/结束按钮结束采样，仍会按原逻辑使用已收集的有效样本求解并保存结果；
样本不足时退出。它不是“丢弃所有样本”。达到目标数量后自动计算并退出。

悬停验证获得参考标记位姿后会切换成原始彩色画面，保持等待终端确认和运动期间的
网页监控；任务结束后服务关闭。初始稳定检测通常很快，dry-run 可能在打开网页前
就计算完成并退出，长时间查看画面可使用独立 ArUco 检测入口。

## 网页参数

所有四个入口都支持以下参数；标定入口把参数放在 `intrinsic/extrinsic` 后：

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `--web-host` | `0.0.0.0` | 监听所有 IPv4 网卡；可改为 `127.0.0.1` 配合 SSH 转发 |
| `--web-port` | `8080` | 网页端口；端口被占用时应换端口或结束旧进程 |
| `--web-fps` | `10` | 网页最高发布帧率，取值大于 0 且不超过 60 |
| `--web-quality` | `80` | JPEG 质量，取值 1–100 |
| `--no-web` | 关闭 | 加此参数后禁用网页，仅保留终端输出；标定将无法通过网页采样 |

例如带宽有限时：

```bash
python3 scripts/main_pbvs.py --web-port 8081 --web-fps 5 --web-quality 60
```

直接访问模式面向可信局域网，服务没有登录认证。网页请求令牌用于阻止跨站表单
误触发按钮，不是用户身份认证。不应将此端口直接暴露到公网。
需要经加密连接访问时，可让服务只监听回环地址：

```bash
# Jetson 上
python3 scripts/main_pbvs.py --web-host 127.0.0.1

# 电脑上的另一个终端
ssh -N -L 8080:127.0.0.1:8080 jetson@192.168.66.80
```

此时电脑访问 **http://127.0.0.1:8080/**。

## 线程与接口

- 视觉生产者按网页帧率提交图像副本；不调用 JPEG 编码或网络写入。
- 后台线程编码 JPEG，HTTP 客户端线程发送 MJPEG。原始图像和 JPEG 各保留最新一份，
  不积累历史队列，浏览器速度慢时跳过旧帧。
- 网页帧率不修改相机采集频率或 ServoJ 控制频率；仍会消耗一定 CPU 和网络带宽，
  应结合主程序原有周期统计在 Jetson 上确认负载。
- 浏览器断线或关闭标签页不会停止机器人程序；重新打开页面可继续查看。
- 网页每秒检查最后一帧的服务端更新时间，超过 2 秒或服务断线时提示画面过期。
  这不是端到端画面延迟测量，不能把网页图像当作硬件实时反馈。

| HTTP 接口 | 功能 |
| --- | --- |
| `GET /` | 监控页面 |
| `GET /stream.mjpg` | MJPEG 视频流 |
| `GET /status` | 最近编码帧的序号、更新时间、编码错误、结束状态 |
| `POST /command/sample` | 标定采样，需要页面中的请求令牌且当前入口允许采样 |
| `POST /command/quit` | 请求走当前脚本的结束流程，需要页面中的请求令牌 |

## 结束行为

网页结束按钮等价于脚本的结束请求，**不是急停**。PBVS 主程序保留原有
`ServoMoveEnd` 和回到 `[90, -90, 90, -90, -90, 0]` 的退出动作，执行模式下结束仍可能
触发回位。硬件急停仍由现场装置负责。正在阻塞的相机/SDK调用完成后才能处理部分
退出步骤；等待 `input()` 时还需要在 SSH 终端结束输入，网页结束状态会阻止随后确认启动。

悬停测试原有 `finally` 无条件使能/回位已修正：dry-run、取消确认和检测失败均不会
触发该回位；只有确认进入执行阶段、且没有网页结束请求时才运行原有收尾回位。

## 验证方法和范围

无需硬件的网页自检：

```bash
python3 scripts/algorithm/web_preview.py
```

浏览器应看到持续变化的 `TEST` 时间文字，网页结束后进程退出。
这个入口不导入 RealSense 或机器人 SDK。

离线回归测试：

```bash
python3 -m unittest discover -s tests -v
```

测试包含实际 HTTP/MJPEG 传输和 JPEG 解码、多客户端断线、采样/结束事件、命令约束、
慢编码时覆盖旧帧、编码失败恢复、端口释放、相机启动失败退出、标定采样、悬停测试
dry-run/取消/检测失败/网页结束时不发运动指令，以及无 HighGUI 调用检查。
相机和运动学调用点采用替身，不启动真实相机或机器人。

真实 RealSense 画面、Jetson 局域网访问和实机控制周期仍需在现场确认；本次未执行
机械臂运动。连接失败时优先查看启动终端的端口占用、相机占用和检测错误。

本次本机验证结果：12 项离线测试全部通过（Python 3.12、OpenCV 4.6.0），
7 个相关 Python 文件语法检查通过，`git diff --check` 通过。此结果不代表
已验证 Jetson 上的相机驱动、ArUco 检测兼容性或实机时序。
