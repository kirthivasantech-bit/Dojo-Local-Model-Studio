from pathlib import Path

_CACHE: dict = {}


def load_llama(
    path: str,
    n_ctx: int = 4096,
    n_gpu_layers: int = -1,
    n_threads: int | None = None,
):
    if path in _CACHE:
        return _CACHE[path]

    try:
        from llama_cpp import Llama
    except ImportError:
        raise RuntimeError(
            "llama-cpp-python is not installed. "
            "Run: pip install llama-cpp-python"
        )

    if not Path(path).exists():
        raise FileNotFoundError(f"Model file not found: {path}")

    kwargs = {
        "model_path": path,
        "n_ctx": n_ctx,
        "n_gpu_layers": n_gpu_layers,
        "verbose": False,
    }
    if n_threads:
        kwargs["n_threads"] = n_threads

    model = Llama(**kwargs)
    _CACHE[path] = model
    return model


def unload_model(path: str) -> None:
    _CACHE.pop(path, None)


def unload_all() -> None:
    _CACHE.clear()


def stream_chat(
    model,
    messages: list[dict],
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 40,
    repeat_penalty: float = 1.1,
    max_tokens: int = 512,
    seed: int = -1,
):
    stream = model.create_chat_completion(
        messages=messages,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repeat_penalty=repeat_penalty,
        max_tokens=max_tokens,
        seed=seed,
        stream=True,
    )
    for chunk in stream:
        delta = chunk["choices"][0].get("delta", {})
        content = delta.get("content")
        if content:
            yield content


def complete_once(
    model,
    messages: list[dict],
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 40,
    repeat_penalty: float = 1.1,
    max_tokens: int = 512,
    seed: int = -1,
) -> str:
    out = model.create_chat_completion(
        messages=messages,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repeat_penalty=repeat_penalty,
        max_tokens=max_tokens,
        seed=seed,
    )
    return out["choices"][0]["message"]["content"]
def stream_complete(
    model,
    messages: list[dict],
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 40,
    repeat_penalty: float = 1.1,
    max_tokens: int = 512,
    seed: int = -1,
):
    """Stream tokens from a single-shot completion (no chat history applied)."""
    stream = model.create_chat_completion(
        messages=messages,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repeat_penalty=repeat_penalty,
        max_tokens=max_tokens,
        seed=seed,
        stream=True,
    )
    for chunk in stream:
        delta = chunk["choices"][0].get("delta", {})
        content = delta.get("content")
        if content:
            yield content