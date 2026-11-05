# Jetson PBVS 依赖冲突修复（2026-10-22）

环境：Ubuntu 22.04.5、aarch64、Python 3.10.12，使用系统 `python3` 及用户 site-packages。

## 根因与修复

`python3 scripts/main_pbvs.py --help` 即可复现：当前加载 NumPy 2.2.6，
现有 OpenCV 4.10.0 二进制按 NumPy 1.x ABI 编译，导入时报
`AttributeError: _ARRAY_API not found` / `ImportError: numpy.core.multiarray failed to import`。
此类错误及降级 NumPy 的处理方式见 [NumPy 官方排查文档](https://numpy.org/doc/stable/user/troubleshooting-importerror.html)。

机器人库间接依赖 `pgraph-python 1.0.0`，其元数据要求 `numpy>=2.0`。
所以只降级 NumPy 仍留下依赖冲突，后续安装可能重新升级 NumPy。

在 Jetson 实际执行：

```bash
python3 -m pip freeze > /tmp/fr_visual_servo-pip-before-20261022.txt
python3 -m pip install --user 'numpy==1.26.4' 'pgraph-python==0.6.5'
```

其他运行库沿用已有版本；已将本次验证的核心依赖固定到项目根目录的
`requirements-jetson.txt`，包含正确的发行包名 `roboticstoolbox-python`。

```bash
# 重建核心运行依赖
python3 -m pip install --user -r requirements-jetson.txt

# 添加其他包时，约束本项目核心依赖
python3 -m pip install -c requirements-jetson.txt PACKAGE_NAME
```

版本：NumPy 1.26.4、pgraph-python 0.6.5、opencv-contrib-python 4.10.0.84、
SciPy 1.15.1、Matplotlib 3.10.6、spatialmath-python 1.1.18、
spatialgeometry 1.4.2、roboticstoolbox-python 1.4.4、pyrealsense2 2.59.1.11053。
这是核心运行依赖约束，并非整个 Jetson 环境的完整锁文件。

## 实测验证

- NumPy、OpenCV、SciPy、spatialmath、spatialgeometry、roboticstoolbox、pgraph、
  pyrealsense2 均可导入；主程序 `--help` 成功。
- `python3 scripts/camera_preview.py --camera realsense --frames 30 --no-web`：
  `640x480@30`，采集 30 帧，用时 1.42 秒。
- 下列 dry-run 连续运行约 35 秒，连接 `192.168.58.2`、读取实际关节、
  检测真实 ArUco 标记，完成 FK/IK 与关节限位检查，共输出 2420 个控制周期。
  状态从 `servoing` 切换到 `blind-approach`，悬停参数为 400 mm / 40 mm。
  无 traceback 或警告；收到 SIGINT 后相机线程和主程序正常退出。

```bash
timeout -s INT -k 8s 35s python3 -u scripts/main_pbvs.py \
  --camera realsense --mode tool1 --hover-far 0.4 --hover-near 0.04 \
  --blind-final-approach
```

- `python3 -m unittest discover -s tests -v`：21 项通过。
- `python3 -m pip install --dry-run --user -r requirements-jetson.txt` 返回 0，
  所有核心依赖已满足，无需新增安装；逐项检查已安装版本均与文件一致。
- 用户原命令（含 `--execute`）已实测到达安全确认提示，终端选择 `1` 使用 RealSense。
  验证时回答 `no`，没有使能或下发运动指令；实际运动与到位效果未验证。

Jetson 临时日志：`/tmp/fr_visual_servo-dry-run-20261022.log`、
`/tmp/fr_visual_servo-tests-20261022.log`、`/tmp/fr_visual_servo-pip-check-20261022.txt`。

## 运行

```bash
cd ~/fairino_ws/fr_visual_servo
python3 scripts/main_pbvs.py --camera realsense --mode tool1 \
  --hover-far 0.4 --hover-near 0.04 --blind-final-approach --execute
```

不加 `--camera realsense` 时仍按原逻辑在终端选择 `1`。
网页监控地址为 `http://192.168.66.80:8080/`，现场确认安全后才在终端回答 `yes`。

## 全局环境剩余提示

修复后 `pip check` 已无 NumPy / pgraph 冲突，但系统其他用途的包仍有既有提示：
Orbbec 示例依赖 av/open3d/pynput 缺失、tensorflowjs 的 packaging 约束、
httpcore 的 h11 约束，以及系统 PyNaCl 平台标记提示。
这些未阻碍本次 RealSense PBVS 实测，未在此次修复中安装或调整。
