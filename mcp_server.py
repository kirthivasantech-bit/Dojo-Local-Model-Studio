"""
Agent Dojo MCP Server

Lets other AI apps (Claude, Cursor, Windsurf) use your local models
as tools. Run with: mcp dev mcp_server.py
"""

import json
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from core.downloader import DOWNLOAD_MANAGER
from core.eval import load_llama, complete_once
from core.registry import (
    PRESET_MODELS,
    build_download_url,
    delete_model,
    disk_usage,
    list_local_models,
    list_repo_ggufs,
    search_gguf_models,
)
from core.sessions import load_sessions


mcp = FastMCP("Agent Dojo")


# ------------------------------------------------------------------
# TOOLS — actions other apps can ask Agent Dojo to perform
# ------------------------------------------------------------------

@mcp.tool()
def list_models() -> list[dict]:
    """List every GGUF model downloaded on this computer.
    Use this first to see what is available to chat with."""
    return list_local_models()


@mcp.tool()
def search_models(query: str, limit: int = 10) -> list[dict]:
    """Search Hugging Face for GGUF model repositories.

    Args:
        query: Search term like 'qwen', 'llama', or 'mistral'.
        limit: Max results to return.
    """
    return search_gguf_models(query, limit=limit)


@mcp.tool()
def list_model_files(repo_id: str) -> list[dict]:
    """List the .gguf files available in a Hugging Face repo.

    Args:
        repo_id: Full repo ID like 'Qwen/Qwen2.5-0.5B-Instruct-GGUF'.
    """
    return list_repo_ggufs(repo_id)


@mcp.tool()
def download_model(
    repo_id: str, filename: str, hf_token: str | None = None
) -> dict:
    """Download a GGUF model from Hugging Face to this machine.

    Args:
        repo_id: Hugging Face repository ID.
        filename: The .gguf filename to download.
        hf_token: Optional token for private repos.
    """
    url = build_download_url(repo_id, filename)
    task = DOWNLOAD_MANAGER.get_or_create(
        repo_id, filename, url, hf_token=hf_token,
        name=Path(filename).stem,
    )
    if task.status in ("pending", "failed"):
        task.start()
    return {
        "status": task.status,
        "message": "Download started.",
    }


@mcp.tool()
def chat(
    model_path: str,
    message: str,
    system_prompt: str = "You are a helpful assistant.",
    temperature: float = 0.7,
    max_tokens: int = 512,
) -> str:
    """Send a message to a local model and get the response.

    The model runs entirely on this machine. Nothing leaves the device.

    Args:
        model_path: Full path to the .gguf file, from list_models.
        message: The question or request for the model.
        system_prompt: Sets the persona.
        temperature: 0 is precise, higher is creative.
        max_tokens: Maximum response length.
    """
    model = load_llama(model_path)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": message},
    ]
    return complete_once(
        model, messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )


@mcp.tool()
def delete_local_model(model_path: str) -> dict:
    """Delete a downloaded model to free disk space.

    Args:
        model_path: Full path to the .gguf file, from list_models.
    """
    p = Path(model_path)
    if not p.exists():
        return {"error": f"Not found: {model_path}"}
    size_gb = round(p.stat().st_size / 1e9, 2)
    delete_model(model_path)
    return {"deleted": model_path, "freed_gb": size_gb}


@mcp.tool()
def get_storage_info() -> dict:
    """Report how much disk space the model library uses."""
    return disk_usage()


@mcp.tool()
def list_preset_models() -> list[dict]:
    """Show the recommended starter models with sizes."""
    return PRESET_MODELS


# ------------------------------------------------------------------
# RESOURCES — read-only data apps can look at
# ------------------------------------------------------------------

@mcp.resource("agentdojo://models")
def resource_models() -> str:
    """The list of downloaded models."""
    return json.dumps(list_local_models(), indent=2)


@mcp.resource("agentdojo://storage")
def resource_storage() -> str:
    """Disk usage of the model library."""
    return json.dumps(disk_usage(), indent=2)


@mcp.resource("agentdojo://sessions")
def resource_sessions() -> str:
    """Summary of saved chat sessions."""
    sessions = load_sessions()
    summary = {}
    for sid, sess in sessions.items():
        summary[sid] = {
            "title": sess.get("title"),
            "model": sess.get("model_name"),
            "messages": len(sess.get("messages", [])),
        }
    return json.dumps(summary, indent=2)


# ------------------------------------------------------------------
# PROMPTS — pre-built templates
# ------------------------------------------------------------------

@mcp.prompt()
def compare_models(question: str) -> str:
    """Ask two local models the same question and compare."""
    return (
        f"Use list_models, then ask this question to two different "
        f"models using the chat tool, and compare their answers:\n\n"
        f"{question}"
    )


if __name__ == "__main__":
    mcp.run()