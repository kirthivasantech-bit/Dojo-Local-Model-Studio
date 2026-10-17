import json
import time
import uuid
from pathlib import Path

SESSIONS_FILE = Path("sessions.json")


DEFAULT_SETTINGS = {
    "system_prompt": "You are a helpful assistant.",
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 40,
    "repeat_penalty": 1.1,
    "max_tokens": 512,
    "seed": -1,
}


def load_sessions() -> dict:
    if not SESSIONS_FILE.exists():
        return {}
    try:
        data = json.loads(SESSIONS_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}
    for sess in data.values():
        for k, v in DEFAULT_SETTINGS.items():
            sess.setdefault(k, v)
    return data


def save_sessions(sessions: dict) -> None:
    try:
        SESSIONS_FILE.write_text(
            json.dumps(sessions, indent=2), encoding="utf-8"
        )
    except Exception:
        pass


def new_session() -> tuple[str, dict]:
    sid = uuid.uuid4().hex[:10]
    return sid, {
        "id": sid,
        "title": "New chat",
        "model_path": None,
        "model_name": None,
        "created_at": time.time(),
        "updated_at": time.time(),
        "messages": [],
        **DEFAULT_SETTINGS,
    }


def title_from_message(text: str, maxlen: int = 42) -> str:
    text = " ".join(text.strip().split())
    if not text:
        return "New chat"
    if len(text) <= maxlen:
        return text
    return text[: maxlen - 1] + "\u2026"


def export_session_markdown(sess: dict) -> str:
    lines = [f"# {sess['title']}", ""]
    if sess.get("model_name"):
        lines.append(f"Model: {sess['model_name']}")
        lines.append("")
    lines.append("---")
    lines.append("")
    for msg in sess["messages"]:
        role = msg["role"].capitalize()
        lines.append(f"### {role}")
        lines.append("")
        lines.append(msg["content"])
        lines.append("")
    return "\n".join(lines)