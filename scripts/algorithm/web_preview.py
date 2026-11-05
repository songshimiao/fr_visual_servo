"""Headless MJPEG preview. HTTP and JPEG encoding never run in the servo loop."""

import argparse
import html
import json
import secrets
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

import cv2


def add_web_arguments(parser):
    parser.add_argument("--web-host", default="0.0.0.0", help="网页监听地址，默认所有网卡")
    parser.add_argument("--web-port", type=int, default=8080, help="网页端口，默认8080")
    parser.add_argument("--web-fps", type=float, default=10.0, help="网页最高帧率，不改变相机/伺服频率")
    parser.add_argument("--web-quality", type=int, default=80, help="JPEG质量，1~100")
    parser.add_argument("--no-web", action="store_true", help="关闭网页，仅保留终端输出")


class WebPreview:
    """One raw-frame slot and one JPEG slot; slow clients cannot queue frames.

    publish() is called by the camera producer, not the robot control loop.
    A quit request latches stop_event; sample requests coalesce into one event.
    The HTTP API never sends robot commands or enables robot execution.
    """

    def __init__(self, host="0.0.0.0", port=8080, fps=10.0, quality=80,
                 title="Camera preview", allow_capture=False, stop_event=None,
                 enabled=True):
        if not 0 <= port <= 65535:
            raise ValueError("web-port must be between 0 and 65535")
        if not 0 < fps <= 60:
            raise ValueError("web-fps must be greater than 0 and at most 60")
        if not 1 <= quality <= 100:
            raise ValueError("web-quality must be between 1 and 100")
        self.host, self.port = host, port
        self.fps, self.quality = fps, quality
        self.title, self.allow_capture, self.enabled = title, allow_capture, enabled
        self.stop_event = stop_event if stop_event is not None else threading.Event()
        self._sample = threading.Event()
        self._closed = threading.Event()
        self._condition = threading.Condition()
        self._raw = None
        self._jpeg = None
        self._sequence = 0
        self._captured_at = None
        self._last_offer = float("-inf")
        self._error = None
        self._token = secrets.token_urlsafe(24)
        self._server = self._http_thread = self._encoder_thread = None

    @classmethod
    def from_args(cls, args, **kwargs):
        return cls(host=args.web_host, port=args.web_port, fps=args.web_fps,
                   quality=args.web_quality, enabled=not args.no_web, **kwargs)

    def __enter__(self):
        return self.start()

    def __exit__(self, *_):
        self.close()

    def start(self):
        if self._server is not None or not self.enabled:
            return self
        self._server = ThreadingHTTPServer((self.host, self.port), self._handler())
        self._server.daemon_threads = True
        self.port = self._server.server_address[1]
        self._http_thread = threading.Thread(target=self._server.serve_forever,
                                             kwargs={"poll_interval": 0.1}, daemon=True)
        self._encoder_thread = threading.Thread(target=self._encode_loop, daemon=True)
        self._http_thread.start()
        self._encoder_thread.start()
        address = "<Jetson-IP>" if self.host == "0.0.0.0" else self.host
        print(f"[网页预览] http://{address}:{self.port}/  ({self.title})")
        return self

    def publish(self, frame):
        if not self.enabled or self._closed.is_set():
            return
        now = time.monotonic()
        if now - self._last_offer < 1.0 / self.fps:
            return
        self._last_offer = now
        # Copy only at the preview rate. Encoding and socket writes are asynchronous.
        snapshot = frame.copy()
        with self._condition:
            self._raw = (snapshot, now)
            self._condition.notify_all()

    def poll_key(self):
        if self.stop_event.is_set():
            return ord("q")
        if self._sample.is_set():
            self._sample.clear()
            return ord("s")
        return -1

    def _encode_loop(self):
        while not self._closed.is_set():
            with self._condition:
                self._condition.wait_for(lambda: self._raw is not None or self._closed.is_set())
                if self._closed.is_set():
                    return
                frame, captured_at = self._raw
                self._raw = None
            try:
                ok, encoded = cv2.imencode(".jpg", frame,
                                          [cv2.IMWRITE_JPEG_QUALITY, self.quality])
                if not ok:
                    raise RuntimeError("JPEG encoder returned no image")
                jpeg = encoded.tobytes()
            except Exception as exc:
                with self._condition:
                    self._error = str(exc)
                continue
            with self._condition:
                self._jpeg = jpeg
                self._captured_at = captured_at
                self._sequence += 1
                self._error = None
                self._condition.notify_all()

    def status(self):
        with self._condition:
            return {"title": self.title, "frame_sequence": self._sequence,
                    "frame_age_s": None if self._captured_at is None else
                    round(time.monotonic() - self._captured_at, 2),
                    "stopping": self.stop_event.is_set(), "error": self._error,
                    "capture_enabled": self.allow_capture}

    def _page(self):
        capture = ('<button onclick="command(\'sample\')">采集一帧 (s)</button>'
                   if self.allow_capture else "")
        page = """<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title><style>
body{font-family:system-ui,sans-serif;max-width:1000px;margin:24px auto;padding:0 16px;background:#121923;color:#eef3fa}
img{display:block;width:100%;height:auto;min-height:180px;background:#05090d;border-radius:8px}
button{padding:12px 18px;margin:12px 12px 0 0;cursor:pointer}#status{padding:12px 0}
.bad{color:#ffad8f}small{color:#bcc9da}</style>
<h1>__TITLE__</h1><div id="status">等待第一帧…</div>
<img id="video" src="/stream.mjpg" alt="相机视频流">
__CAPTURE__<button onclick="command('quit')">结束当前程序/采集 (q)</button>
<p><small>结束按钮触发脚本原有退出流程，可能包含机械臂回位；不是急停。
机械臂执行确认仍在 SSH 终端完成。关闭浏览器不会停止程序。</small></p>
<p id="message"></p><script>
const token=__TOKEN__;
async function command(action){
 try{const r=await fetch('/command/'+action,{method:'POST',headers:{'X-Preview-Token':token}});
 document.getElementById('message').textContent=r.ok?(action==='sample'?'采样请求已提交，以终端采集结果为准。':'结束请求已提交。'):'操作失败：'+r.status;
 }catch(e){document.getElementById('message').textContent='连接已断开。';}}
const statusEl=document.getElementById('status');
async function poll(){try{const r=await fetch('/status',{cache:'no-store'});if(!r.ok)throw Error();const s=await r.json();
 const stale=s.frame_age_s===null||s.frame_age_s>2;statusEl.className=stale||s.stopping||s.error?'bad':'';
 statusEl.textContent=s.stopping?'已请求结束':s.error?'编码错误：'+s.error:s.frame_age_s===null?'等待相机画面…':(stale?'画面未更新':'视频正常')+' | 第 '+s.frame_sequence+' 帧 | 距更新 '+s.frame_age_s+' 秒';
 }catch(e){statusEl.className='bad';statusEl.textContent='服务已断开，当前画面可能已过期。';}}
setInterval(poll,1000);poll();
document.addEventListener('keydown',e=>{if(e.repeat||/INPUT|TEXTAREA/.test(e.target.tagName))return;
 if(e.key==='q')command('quit');if(e.key==='s'&&__ALLOW_CAPTURE__)command('sample');});
</script></html>"""
        return (page.replace("__TITLE__", html.escape(self.title))
                .replace("__CAPTURE__", capture)
                .replace("__TOKEN__", json.dumps(self._token))
                .replace("__ALLOW_CAPTURE__", json.dumps(self.allow_capture))).encode("utf-8")

    def _handler(self):
        preview = self

        class Handler(BaseHTTPRequestHandler):
            def setup(self):
                super().setup()
                self.connection.settimeout(3.0)

            def log_message(self, *_):
                pass

            def reply(self, code, data, content_type="application/json; charset=utf-8"):
                try:
                    self.send_response(code)
                    self.send_header("Content-Type", content_type)
                    self.send_header("Content-Length", str(len(data)))
                    self.send_header("Cache-Control", "no-store")
                    self.send_header("X-Content-Type-Options", "nosniff")
                    self.send_header("X-Frame-Options", "DENY")
                    self.end_headers()
                    self.wfile.write(data)
                except (OSError, TimeoutError):
                    # Browsers may cancel status/page requests while the SDK starts.
                    self.close_connection = True

            def do_GET(self):
                path = urlsplit(self.path).path
                if path == "/":
                    self.reply(200, preview._page(), "text/html; charset=utf-8")
                elif path == "/status":
                    self.reply(200, json.dumps(preview.status()).encode())
                elif path == "/stream.mjpg":
                    self.send_response(200)
                    self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    sequence = 0
                    try:
                        while not preview._closed.is_set():
                            with preview._condition:
                                ready = preview._condition.wait_for(
                                    lambda: preview._sequence != sequence or preview._closed.is_set(),
                                    timeout=1.0)
                                if preview._closed.is_set():
                                    break
                                if not ready:
                                    continue
                                sequence, jpeg = preview._sequence, preview._jpeg
                            self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                             + str(len(jpeg)).encode() + b"\r\n\r\n" + jpeg + b"\r\n")
                            self.wfile.flush()
                    except (OSError, TimeoutError):
                        pass
                else:
                    self.reply(404, b'{"error":"not found"}')

            def do_POST(self):
                # A per-process token prevents cross-origin forms from issuing commands.
                if self.headers.get("X-Preview-Token") != preview._token:
                    self.reply(403, b'{"error":"invalid preview token"}')
                    return
                path = urlsplit(self.path).path
                if path == "/command/quit":
                    preview.stop_event.set()
                elif path == "/command/sample" and preview.allow_capture:
                    preview._sample.set()
                else:
                    self.reply(404, b'{"error":"unsupported command"}')
                    return
                self.reply(200, b'{"ok":true}')

        return Handler

    def close(self):
        self._closed.set()
        self.stop_event.set()
        with self._condition:
            self._condition.notify_all()
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
        for thread in (self._http_thread, self._encoder_thread):
            if thread is not None:
                thread.join(timeout=3.0)


def main():
    """Synthetic camera feed for testing HTTP/JPEG without RealSense or a robot."""
    import numpy as np

    parser = argparse.ArgumentParser(description="无硬件网页视频流自检")
    add_web_arguments(parser)
    args = parser.parse_args()
    with WebPreview.from_args(args, title="Synthetic preview (no hardware)") as preview:
        try:
            while preview.poll_key() != ord("q"):
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(frame, f"TEST {time.monotonic():.2f}", (30, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                preview.publish(frame)
                time.sleep(0.02)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
