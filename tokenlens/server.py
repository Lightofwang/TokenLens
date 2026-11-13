from __future__ import annotations

import argparse
import csv
import io
import json
import mimetypes
import os
import shutil
import socket
import subprocess
import sys
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .config import load_config, save_config
from .pricing import MODEL_PRICING
from .scanner import TokenScanner

STATIC_DIR = Path(__file__).resolve().parent / "static"
SCANNER = TokenScanner()


def _parse_agents_param(qs: dict[str, list[str]]) -> list[str] | None:
    if "agents" not in qs:
        return None
    raw = qs["agents"][0]
    if raw == "__none__":
        return []
    if not raw.strip():
        return None
    return [x.strip() for x in raw.split(",") if x.strip()]


class TokenLensHandler(BaseHTTPRequestHandler):
    server_version = "TokenLens/0.1"

    def log_message(self, format: str, *args) -> None:  # noqa: A002
        return

    def _send_json(self, payload: object, status: int = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path: Path) -> None:
        if not path.exists() or not path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND, "Not Found")
            return
        mime, _ = mimetypes.guess_type(str(path))
        if path.suffix == ".js":
            mime = "application/javascript"
        elif path.suffix == ".css":
            mime = "text/css"
        data = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", f"{mime or 'application/octet-stream'}; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path
        qs = parse_qs(parsed.query, keep_blank_values=True)

        if path == "/api/overview":
            agents = _parse_agents_param(qs)
            range_type = qs.get("range", ["7d"])[0]
            date_str = qs.get("date", [None])[0]
            start_str = qs.get("start", [None])[0]
            end_str = qs.get("end", [None])[0]
            granularity = qs.get("granularity", ["auto"])[0]
            query = qs.get("q", [""])[0]
            data = SCANNER.build_overview(
                selected_agents=agents,
                range_type=range_type,
                date_str=date_str,
                start_str=start_str,
                end_str=end_str,
                granularity=granularity,
                query=query,
            )
            self._send_json(data)
            return

        if path == "/api/sources":
            self._send_json({"sources": SCANNER.get_sources()})
            return

        if path == "/api/session":
            agent_id = qs.get("agent_id", [""])[0]
            session_id = qs.get("session_id", [""])[0]
            detail = SCANNER.get_session_detail(agent_id, session_id)
            if detail is None:
                self._send_json({"error": "Session not found"}, status=HTTPStatus.NOT_FOUND)
            else:
                self._send_json(detail)
            return

        if path == "/api/config":
            self._send_json({"config": load_config(), "default_pricing": MODEL_PRICING})
            return

        if path == "/api/export":
            fmt = qs.get("format", ["json"])[0].lower()
            agents = _parse_agents_param(qs)
            range_type = qs.get("range", ["7d"])[0]
            date_str = qs.get("date", [None])[0]
            start_str = qs.get("start", [None])[0]
            end_str = qs.get("end", [None])[0]
            rows = SCANNER.export_events(agents, range_type, date_str, start_str, end_str)
            if fmt == "csv":
                buf = io.StringIO()
                fieldnames = [
                    "timestamp", "agent_id", "agent_name", "session_id", "model",
                    "uncached_input", "cache_read", "cache_write", "output",
                    "total_tokens", "credits", "is_estimated", "title", "project_path",
                ]
                writer = csv.DictWriter(buf, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
                body = buf.getvalue().encode("utf-8-sig")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/csv; charset=utf-8")
                self.send_header("Content-Disposition", f'attachment; filename="tokenlens-{range_type}.csv"')
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            else:
                body = json.dumps(rows, ensure_ascii=False, indent=2).encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Disposition", f'attachment; filename="tokenlens-{range_type}.json"')
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            return

        # Static files
        if path in ("/", "/index.html"):
            self._send_file(STATIC_DIR / "index.html")
            return
        rel = path.lstrip("/")
        target = (STATIC_DIR / rel).resolve()
        if STATIC_DIR.resolve() in target.parents and target.exists():
            self._send_file(target)
        else:
            self.send_error(HTTPStatus.NOT_FOUND, "Not Found")

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/api/rescan":
            SCANNER.ensure_scanned(force=True)
            self._send_json({"ok": True, "sources": SCANNER.get_sources()})
            return

        if parsed.path == "/api/config":
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length).decode("utf-8", errors="ignore")
            try:
                body = json.loads(raw) if raw else {}
            except Exception:
                self._send_json({"error": "Invalid JSON"}, status=HTTPStatus.BAD_REQUEST)
                return
            cfg = save_config(body)
            self._send_json({"ok": True, "config": cfg})
            return

        self.send_error(HTTPStatus.NOT_FOUND, "Not Found")


def _find_free_port(preferred: int = 7890) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        if s.connect_ex(("127.0.0.1", preferred)) != 0:
            return preferred
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _open_desktop_window(url: str) -> None:
    """Try opening in native-like standalone desktop app window across Windows/macOS/Linux."""
    candidates: list[str] = []
    if os.name == "nt":
        for p in (
            r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
            r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
            r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        ):
            if os.path.exists(p):
                candidates.append(p)
    elif sys.platform == "darwin":
        mac_apps = (
            "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
            "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
            "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser",
            "/Applications/Chromium.app/Contents/MacOS/Chromium",
        )
        for p in mac_apps:
            if os.path.exists(p):
                candidates.append(p)
            user_app = Path.home() / p.lstrip("/")
            if user_app.exists():
                candidates.append(str(user_app))
    for name in ("msedge", "google-chrome", "chromium", "chrome"):
        found = shutil.which(name)
        if found:
            candidates.append(found)

    for browser_exe in candidates:
        try:
            subprocess.Popen(
                [browser_exe, f"--app={url}", "--window-size=1420,900"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return
        except Exception:
            continue
    webbrowser.open(url)


def main() -> None:
    parser = argparse.ArgumentParser(description="TokenLens - Local Multi-Agent Token Dashboard")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=0, help="Bind port (default: auto, prefer 7890)")
    parser.add_argument("--no-open", action="store_true", help="Do not automatically open UI window")
    parser.add_argument("--browser", action="store_true", help="Open in normal browser tab instead of app window")
    args = parser.parse_args()

    port = args.port if args.port > 0 else _find_free_port(7890)
    SCANNER.ensure_scanned()
    server = ThreadingHTTPServer((args.host, port), TokenLensHandler)
    url = f"http://{args.host}:{port}"
    print(f"TokenLens running at {url}")

    if not args.no_open:
        if args.browser:
            webbrowser.open(url)
        else:
            _open_desktop_window(url)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nTokenLens stopped.")
        server.server_close()
