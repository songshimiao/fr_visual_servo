#!/usr/bin/env bash
# Install only the camera runtime; preserve the project's working OpenCV/NumPy.
set -euo pipefail
CAMERA_PYTHON="${CAMERA_PYTHON:-python3}"
if [[ "$(uname -s)" != Linux || "$(uname -m)" != aarch64 ]]; then
    echo '本脚本用于 Jetson Linux ARM64，请直接在 Jetson 上运行。' >&2
    exit 1
fi
"$CAMERA_PYTHON" - <<'PY'
import cv2, numpy, sys
if not ((3, 9) <= sys.version_info[:2] <= (3, 13)):
    raise SystemExit('需要 Python 3.9–3.13；Python 3.8 请使用官方 Release wheel')
if not hasattr(cv2, 'aruco'):
    raise SystemExit('当前 OpenCV 缺少 aruco，请使用工程已有的 opencv-contrib 环境')
print('Python:', sys.version.split()[0], 'OpenCV:', cv2.__version__, 'NumPy:', numpy.__version__)
PY
pip_scope=(--user)
if "$CAMERA_PYTHON" -c 'import sys; sys.exit(sys.prefix == sys.base_prefix)'; then
    pip_scope=()  # A virtual environment must not use --user.
fi
"$CAMERA_PYTHON" -m pip install "${pip_scope[@]}" --no-deps --only-binary=:all: pyorbbecsdk2==2.1.2
# SDK imports may write a diagnostic line before the path: ask the interpreter for metadata.
rules_path="$("$CAMERA_PYTHON" - <<'PY'
from importlib.metadata import distribution
print(distribution('pyorbbecsdk2').locate_file('pyorbbecsdk/shared/99-obsensor-libusb.rules'))
PY
)"
test -f "$rules_path"
destination=/etc/udev/rules.d/99-obsensor-libusb.rules
if [[ -f "$destination" ]] && ! cmp -s "$rules_path" "$destination"; then
    sudo cp -a "$destination" "${destination}.backup.$(date +%Y%m%d%H%M%S)"
fi
sudo install -m 0644 "$rules_path" "$destination"
sudo udevadm control --reload-rules
sudo udevadm trigger --subsystem-match=usb --attr-match=idVendor=2bc5
"$CAMERA_PYTHON" - <<'PY'
import cv2, numpy, pyrealsense2, pyorbbecsdk as ob
from importlib.metadata import version
print('pyorbbecsdk2:', version('pyorbbecsdk2'), 'Orbbec SDK:', ob.get_version())
print('OpenCV:', cv2.__version__, 'NumPy:', numpy.__version__, 'RealSense:', version('pyrealsense2'))
context = ob.Context()  # Keep SDK device manager alive during enumeration.
devices = context.query_devices()
print('Orbbec devices:', devices.get_count())
for i in range(devices.get_count()):
    info = devices.get_device_by_index(i).get_device_info()
    print(info.get_name(), info.get_serial_number(), 'FW:', info.get_firmware_version())
PY
echo '环境已配置。如未枚举到相机，请重新插拔 USB 3 线缆后重试。'
