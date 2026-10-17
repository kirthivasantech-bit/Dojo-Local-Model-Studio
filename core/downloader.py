import json
import threading
from datetime import datetime
from pathlib import Path

import requests

MODELS_DIR = Path("models")


class DownloadTask:
    def __init__(self, repo_id, filename, url, hf_token=None, name=None):
        self.repo_id = repo_id
        self.filename = filename
        self.url = url
        self.hf_token = hf_token
        self.name = name or Path(filename).stem

        self.dest_dir = MODELS_DIR / repo_id.replace("/", "--")
        self.dest = self.dest_dir / filename
        self.part = self.dest.with_suffix(self.dest.suffix + ".part")

        self.total = 0
        self.downloaded = 0
        self.status = "pending"
        self.error = None

        self._stop_flag = False
        self._thread = None

    def _headers(self, existing=0):
        h = {"User-Agent": "agent-dojo/1.0"}
        if self.hf_token:
            h["Authorization"] = f"Bearer {self.hf_token}"
        if existing > 0:
            h["Range"] = f"bytes={existing}-"
        return h

    def _run(self):
        try:
            self.dest_dir.mkdir(parents=True, exist_ok=True)

            if self.dest.exists():
                self.status = "done"
                self._write_meta()
                return

            existing = (
                self.part.stat().st_size if self.part.exists() else 0
            )

            r = requests.get(
                self.url,
                headers=self._headers(existing),
                stream=True,
                timeout=30,
                allow_redirects=True,
            )

            if r.status_code == 416:
                if self.part.exists():
                    self.part.rename(self.dest)
                    self.status = "done"
                    self._write_meta()
                    return

            r.raise_for_status()

            content_len = int(r.headers.get("content-length", 0))
            self.total = existing + content_len if content_len else existing
            self.downloaded = existing
            self.status = "running"

            mode = "ab" if existing > 0 else "wb"
            with open(self.part, mode) as f:
                for chunk in r.iter_content(chunk_size=1024 * 256):
                    if self._stop_flag:
                        self.status = "paused"
                        return
                    if chunk:
                        f.write(chunk)
                        self.downloaded += len(chunk)

            self.part.rename(self.dest)
            self._write_meta()
            self.status = "done"

        except Exception as e:
            self.status = "failed"
            self.error = str(e)

    def _write_meta(self):
        meta = {
            "name": self.name,
            "repo_id": self.repo_id,
            "filename": self.filename,
            "downloaded_at": datetime.utcnow().isoformat() + "Z",
        }
        self.dest.with_suffix(".meta.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop_flag = False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def pause(self):
        self._stop_flag = True

    def remove_partial(self):
        self._stop_flag = True
        self.status = "pending"
        self.downloaded = 0
        self.total = 0
        if self.part.exists():
            self.part.unlink()

    def is_active(self):
        return self._thread is not None and self._thread.is_alive()

    def progress(self):
        if self.total <= 0:
            return 0.0
        return min(1.0, self.downloaded / self.total)


class DownloadManager:
    def __init__(self):
        self.tasks: dict[str, DownloadTask] = {}

    def get_or_create(
        self, repo_id, filename, url, hf_token=None, name=None
    ):
        key = f"{repo_id}::{filename}"
        if key not in self.tasks:
            self.tasks[key] = DownloadTask(
                repo_id, filename, url, hf_token, name
            )
        return self.tasks[key]

    def has_active(self) -> bool:
        return any(t.status == "running" for t in self.tasks.values())

    def clear_done(self):
        for key in list(self.tasks.keys()):
            if self.tasks[key].status == "done":
                del self.tasks[key]


DOWNLOAD_MANAGER = DownloadManager()