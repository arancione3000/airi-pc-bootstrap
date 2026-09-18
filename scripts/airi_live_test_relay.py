from __future__ import annotations

import argparse
import json
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse


class Store:
    def __init__(self) -> None:
        self.items: list[dict] = []
        self.cv = threading.Condition()

    def publish(self, topic: str, message: str) -> dict:
        row = {
            "id": uuid.uuid4().hex[:12],
            "time": int(time.time()),
            "event": "message",
            "topic": topic,
            "message": message,
        }
        with self.cv:
            self.items.append(row)
            self.items = self.items[-1000:]
            self.cv.notify_all()
        return row

    def snapshot(self, topic: str) -> list[dict]:
        with self.cv:
            return [x for x in self.items if x["topic"] == topic]


STORE = Store()


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args) -> None:
        return

    def _topic(self) -> str:
        path = urlparse(self.path).path
        parts = [x for x in path.split("/") if x]
        return parts[0] if parts else ""

    def do_POST(self) -> None:  # noqa: N802
        topic = self._topic()
        if not topic:
            self.send_error(404)
            return
        length = int(self.headers.get("content-length", "0") or "0")
        message = self.rfile.read(length).decode("utf-8", errors="replace")
        row = STORE.publish(topic, message)
        body = json.dumps(row, separators=(",", ":")).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        parts = [x for x in parsed.path.split("/") if x]
        if len(parts) != 2 or parts[1] != "json":
            self.send_error(404)
            return
        topic = parts[0]
        poll = parse_qs(parsed.query).get("poll", ["0"])[0] == "1"
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()

        sent = 0
        try:
            while True:
                rows = STORE.snapshot(topic)
                while sent < len(rows):
                    self.wfile.write((json.dumps(rows[sent], separators=(",", ":")) + "\n").encode())
                    self.wfile.flush()
                    sent += 1
                if poll:
                    return
                with STORE.cv:
                    STORE.cv.wait(timeout=15)
                    # Send a harmless keepalive event that the client ignores.
                    if sent == len(STORE.snapshot(topic)):
                        keep = {"id": uuid.uuid4().hex[:12], "time": int(time.time()), "event": "keepalive", "topic": topic}
                        self.wfile.write((json.dumps(keep, separators=(",", ":")) + "\n").encode())
                        self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            return


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8765)
    args = p.parse_args()
    server = ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
    print(f"AIRI_TEST_RELAY_READY={args.port}", flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
