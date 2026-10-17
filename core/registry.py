import json
import os
from datetime import datetime
from pathlib import Path

from huggingface_hub import HfApi

MODELS_DIR = Path(os.getenv("MODELS_DIR", "models"))


PRESET_MODELS = [
    {
        "name": "Qwen 2.5 0.5B",
        "repo": "Qwen/Qwen2.5-0.5B-Instruct-GGUF",
        "file": "qwen2.5-0.5b-instruct-q4_k_m.gguf",
        "size": "400 MB",
        "description": "Fastest. Runs on 1 GB RAM.",
    },
    {
        "name": "Qwen 2.5 1.5B",
        "repo": "Qwen/Qwen2.5-1.5B-Instruct-GGUF",
        "file": "qwen2.5-1.5b-instruct-q4_k_m.gguf",
        "size": "1.1 GB",
        "description": "Better quality. Still light.",
    },
    {
        "name": "TinyLlama 1.1B",
        "repo": "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
        "file": "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
        "size": "640 MB",
        "description": "General chat.",
    },
    {
        "name": "Llama 3.2 1B",
        "repo": "bartowski/Llama-3.2-1B-Instruct-GGUF",
        "file": "Llama-3.2-1B-Instruct-Q4_K_M.gguf",
        "size": "800 MB",
        "description": "Meta. Requires HF login.",
    },
]


def build_download_url(repo_id: str, filename: str) -> str:
    return f"https://huggingface.co/{repo_id}/resolve/main/{filename}"


def _meta_path(gguf_path: Path) -> Path:
    return gguf_path.with_suffix(".meta.json")


def list_local_models() -> list[dict]:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    out = []
    for path in sorted(MODELS_DIR.rglob("*.gguf")):
        meta_file = _meta_path(path)
        meta = {}
        if meta_file.exists():
            try:
                meta = json.loads(meta_file.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                meta = {}
        out.append({
            "name": meta.get("name", path.stem),
            "path": str(path),
            "size_gb": round(path.stat().st_size / 1e9, 2),
            "repo_id": meta.get("repo_id"),
            "downloaded_at": meta.get("downloaded_at"),
        })
    return out


def delete_model(path: str) -> None:
    p = Path(path)
    if not p.exists():
        return
    meta = _meta_path(p)
    if meta.exists():
        meta.unlink()
    p.unlink()


def search_gguf_models(query: str, limit: int = 20) -> list[dict]:
    api = HfApi()
    try:
        results = api.list_models(
            search=query,
            filter="gguf",
            limit=limit * 2,
        )
    except Exception as e:
        raise ValueError(f"Search failed: {e}")

    out = []
    for r in results:
        out.append({
            "repo_id": r.id,
            "downloads": getattr(r, "downloads", 0) or 0,
            "likes": getattr(r, "likes", 0) or 0,
        })

    # Sort by downloads, most popular first.
    # Some HF API versions don't support server-side sorting, so we
    # sort locally. Works on every version.
    out.sort(key=lambda x: x["downloads"], reverse=True)
    return out[:limit]


def list_repo_ggufs(
    repo_id: str, hf_token: str | None = None
) -> list[dict]:
    if "/" not in repo_id:
        raise ValueError(
            f"Invalid repo ID: '{repo_id}'. "
            "Use 'organization/model-name'."
        )
    api = HfApi()
    try:
        info = api.model_info(
            repo_id, files_metadata=True, token=hf_token
        )
    except Exception as e:
        raise ValueError(f"Could not read repo '{repo_id}': {e}")

    out = []
    for sib in info.siblings or []:
        name = sib.rfilename
        if not name.endswith(".gguf"):
            continue
        size = getattr(sib, "size", None) or 0
        out.append({
            "filename": name,
            "size_gb": round(size / 1e9, 2) if size else 0,
        })
    if not out:
        raise ValueError(f"Repo '{repo_id}' contains no .gguf files.")
    return sorted(out, key=lambda x: x["filename"])


def disk_usage() -> dict:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    total = 0
    count = 0
    for path in MODELS_DIR.rglob("*.gguf"):
        total += path.stat().st_size
        count += 1
    return {"count": count, "total_gb": round(total / 1e9, 2)}