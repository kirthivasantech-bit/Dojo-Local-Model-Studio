"""
Voice engine for Agent Dojo.

Speech-to-text via faster-whisper (default: tiny.en, fast on CPU).
Text-to-speech via piper-tts (offline, natural voices).
"""

import base64
import io
import os
import tempfile
import wave
from pathlib import Path

import requests

VOICES_DIR = Path(os.getenv("VOICES_DIR", "voices"))


AVAILABLE_VOICES = [
    {
        "name": "Lessac (US English)",
        "id": "en_US-lessac-medium",
        "lang_family": "en",
        "lang_code": "en_US",
        "voice_name": "lessac",
        "quality": "medium",
    },
    {
        "name": "Amy (US English)",
        "id": "en_US-amy-medium",
        "lang_family": "en",
        "lang_code": "en_US",
        "voice_name": "amy",
        "quality": "medium",
    },
    {
        "name": "Ryan (US English, male)",
        "id": "en_US-ryan-medium",
        "lang_family": "en",
        "lang_code": "en_US",
        "voice_name": "ryan",
        "quality": "medium",
    },
    {
        "name": "Alba (British English)",
        "id": "en_GB-alba-medium",
        "lang_family": "en",
        "lang_code": "en_GB",
        "voice_name": "alba",
        "quality": "medium",
    },
]


def _voice_urls(voice_info: dict) -> tuple[str, str]:
    base = (
        f"https://huggingface.co/rhasspy/piper-voices/resolve/main/"
        f"{voice_info['lang_family']}/"
        f"{voice_info['lang_code']}/"
        f"{voice_info['voice_name']}/"
        f"{voice_info['quality']}/"
        f"{voice_info['id']}"
    )
    return f"{base}.onnx", f"{base}.onnx.json"


def list_downloaded_voices() -> list[dict]:
    VOICES_DIR.mkdir(parents=True, exist_ok=True)
    out = []
    for onnx_path in sorted(VOICES_DIR.glob("*.onnx")):
        config_path = Path(str(onnx_path) + ".json")
        if not config_path.exists():
            continue
        out.append({
            "id": onnx_path.stem,
            "path": str(onnx_path),
            "config_path": str(config_path),
            "size_mb": round(onnx_path.stat().st_size / 1e6, 1),
        })
    return out


def download_voice(voice_info: dict) -> Path:
    VOICES_DIR.mkdir(parents=True, exist_ok=True)
    onnx_path = VOICES_DIR / f"{voice_info['id']}.onnx"
    config_path = VOICES_DIR / f"{voice_info['id']}.onnx.json"

    if onnx_path.exists() and config_path.exists():
        return onnx_path

    onnx_url, config_url = _voice_urls(voice_info)

    for url, dest in [(onnx_url, onnx_path), (config_url, config_path)]:
        r = requests.get(url, stream=True, timeout=60)
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 64):
                if chunk:
                    f.write(chunk)
    return onnx_path


def is_voice_downloaded(voice_id: str) -> bool:
    onnx_path = VOICES_DIR / f"{voice_id}.onnx"
    config_path = VOICES_DIR / f"{voice_id}.onnx.json"
    return onnx_path.exists() and config_path.exists()


class VoiceEngine:
    """Lazy speech engine. Whisper and Piper models cached in memory."""

    def __init__(self, whisper_size: str = "tiny.en"):
        self.whisper_size = whisper_size
        self._whisper = None
        self._voices: dict = {}

    def _load_whisper(self):
        if self._whisper is None:
            try:
                from faster_whisper import WhisperModel
            except ImportError:
                raise RuntimeError(
                    "faster-whisper is not installed. "
                    "Run: pip install faster-whisper"
                )
            self._whisper = WhisperModel(
                self.whisper_size,
                device="cpu",
                compute_type="int8",
                cpu_threads=os.cpu_count() or 4,
            )
        return self._whisper

    def _load_piper(self, voice_id: str):
        if voice_id in self._voices:
            return self._voices[voice_id]

        try:
            from piper import PiperVoice
        except ImportError:
            raise RuntimeError(
                "piper-tts is not installed. "
                "Run: pip install piper-tts --only-binary :all:"
            )

        onnx_path = VOICES_DIR / f"{voice_id}.onnx"
        config_path = VOICES_DIR / f"{voice_id}.onnx.json"

        if not onnx_path.exists():
            raise FileNotFoundError(
                f"Voice '{voice_id}' not downloaded."
            )

        if config_path.exists():
            voice = PiperVoice.load(
                str(onnx_path), config_path=str(config_path)
            )
        else:
            voice = PiperVoice.load(str(onnx_path))

        self._voices[voice_id] = voice
        return voice

    def transcribe_bytes(self, audio_bytes: bytes) -> str:
        model = self._load_whisper()

        tmp = tempfile.NamedTemporaryFile(
            delete=False, suffix=".wav"
        )
        try:
            tmp.write(audio_bytes)
            tmp.close()
            # VAD filter skips silence and speeds up transcription
            segments, _ = model.transcribe(
                tmp.name,
                beam_size=1,
                language="en",
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500),
            )
            return " ".join(seg.text for seg in segments).strip()
        finally:
            try:
                os.unlink(tmp.name)
            except OSError:
                pass

    def synthesize_to_bytes(self, text: str, voice_id: str) -> bytes:
        if not text.strip():
            return b""

        voice = self._load_piper(voice_id)

        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav_file:
            if hasattr(voice, "synthesize_wav"):
                voice.synthesize_wav(text, wav_file)
            elif hasattr(voice, "synthesize"):
                voice.synthesize(text, wav_file)
            else:
                raise RuntimeError("Unsupported piper-tts version.")
        return buffer.getvalue()


_ENGINE: VoiceEngine | None = None


def get_engine() -> VoiceEngine:
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = VoiceEngine()
    return _ENGINE


def audio_autoplay_html(audio_bytes: bytes) -> str:
    """HTML snippet that autoplays audio reliably across browsers."""
    b64 = base64.b64encode(audio_bytes).decode("ascii")
    return f"""
    <audio autoplay controls style="width:100%; margin-top:8px;">
      <source src="data:audio/wav;base64,{b64}" type="audio/wav">
    </audio>
    """