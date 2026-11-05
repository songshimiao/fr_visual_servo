"""Raw color preview / finite smoke test / factory intrinsics export. No robot imports."""
import argparse
import json
from pathlib import Path
import time

from algorithm.camera_stream import add_camera_arguments, resolve_camera_arguments, create_camera
from algorithm.web_preview import WebPreview, add_web_arguments


def main():
    parser = argparse.ArgumentParser(description="双相机彩色采集自检和网页预览（不连接机械臂）")
    add_camera_arguments(parser)
    add_web_arguments(parser)
    parser.add_argument("--frames", type=int, default=0, help="有效帧数量达到此值后退出；0 为持续预览")
    parser.add_argument("--export-intrinsics", type=Path, help="导出当前彩色流出厂内参；已有文件不覆盖")
    args = parser.parse_args()
    if args.frames < 0:
        parser.error("--frames 必须 >= 0")
    settings = resolve_camera_arguments(args)
    stream = create_camera(settings)
    count = 0
    start = time.monotonic()
    try:
        if args.export_intrinsics:
            result = stream.get_factory_intrinsics()
            args.export_intrinsics.parent.mkdir(parents=True, exist_ok=True)
            with args.export_intrinsics.open("x", encoding="utf-8") as output:
                json.dump(result, output, ensure_ascii=False, indent=2)
                output.write("\n")
            print(f"[出厂内参] {args.export_intrinsics}；实际精度请以 ChArUco 标定验证")
        with WebPreview.from_args(args, title=f"{settings.backend} color preview") as preview:
            while not preview.stop_event.is_set() and (args.frames == 0 or count < args.frames):
                frame = stream.get_frame()
                if frame is not None:
                    preview.publish(frame)
                    count += 1
            print(f"[采集完成] {count} 帧，耗时 {time.monotonic() - start:.2f} 秒")
    except KeyboardInterrupt:
        print("[退出] Ctrl+C")
    finally:
        stream.stop()


if __name__ == "__main__":
    main()
