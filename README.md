# Agent Dojo

A local-first model studio. Download GGUF models from Hugging Face, chat with them offline, and have full voice conversations with them. Everything runs on your machine. Nothing is sent to any server.

## Features

- Browse and download GGUF models from Hugging Face
- Pause and resume downloads
- Streaming chat with conversation history
- Multiple chat sessions with rename, export, and delete
- System prompt per session
- Advanced generation controls: temperature, top-p, top-k, repeat penalty, max tokens, seed
- Regenerate last response
- Copy last response
- Export any conversation as Markdown
- Delete models to reclaim disk space
- Voice mode: speak to your models and hear them reply, fully offline
- MCP server so other AI tools (Claude Desktop, Cursor, Windsurf) can use your local models
- Mobile-friendly interface
- Fully offline after download

## Requirements

- Python 3.10 or newer
- 8 GB RAM minimum (16 GB recommended for larger models)
- About 5 GB free disk space for models and voices
- Node.js 22.19.0 or newer (only if you use the MCP Inspector)
- A microphone (only for voice mode)

## Installation

Clone the repository and set up a virtual environment.

```bash
git clone https://github.com/kirthivasantech-bit/agent-dojo.git
cd agent-dojo

python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Mac/Linux

pip install --upgrade pip
pip install -r requirements.txt
```

### Windows: llama-cpp-python DLL error

If you get a DLL error on import, install the prebuilt wheel instead:

```bash
pip uninstall llama-cpp-python -y
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu --force-reinstall
```

If the error persists, install the Microsoft Visual C++ Redistributable from:

```
https://aka.ms/vs/17/release/vc_redist.x64.exe
```

Then restart your terminal.

### Windows: piper-tts build error

If `pip install piper-tts` fails to compile, use the prebuilt wheel:

```bash
pip install piper-tts --only-binary :all:
```

If that version is not available for your Python, try:

```bash
pip install piper-tts==1.2.0
```

## Running

```bash
streamlit run app.py
```

Open `http://localhost:8501`.

The first run downloads nothing. You choose what to download from inside the app.

## How to use

### 1. Download a model

Open the **Models** tab. Click Download on a recommended model, or search Hugging Face for any GGUF repository.

Wait for the download. Files are saved to `models/`. You can pause and resume at any time.

### 2. Chat with the model

Open the **Chat** tab. Pick your model in the sidebar. Type a message. Responses stream in.

Sessions are saved automatically. Create a new one with **+ New chat**. Rename, export, or delete them from the session menu.

### 3. Talk to the model

Open the **Voice** tab. First time only, download a voice pack from the **Models → Voices** tab (about 63 MB each).

Then:

1. Pick a model and voice in the sidebar
2. Click the microphone button
3. Speak
4. Click stop
5. The app transcribes, generates a reply, and speaks it back

Everything runs on your CPU. Nothing leaves your machine.

## Recommended models

| Model | Size | Notes |
|---|---|---|
| Qwen 2.5 0.5B | 400 MB | Fastest. Runs on 1 GB RAM. |
| Qwen 2.5 1.5B | 1.1 GB | Better quality. Still light. |
| TinyLlama 1.1B | 640 MB | General chat. |
| Llama 3.2 1B | 800 MB | Meta. Requires Hugging Face login. |

For voice mode, use Qwen 2.5 1.5B or larger. The 0.5B model gives short generic answers.

## Voice mode

Voice mode uses two offline models:

| Component | Library | Model | Size |
|---|---|---|---|
| Speech to text | faster-whisper | Whisper tiny.en | ~75 MB |
| Text to speech | piper-tts | Lessac, Amy, Ryan, Alba | ~63 MB each |

Whisper downloads automatically the first time you record. Voices are downloaded manually from the Models tab.

Everything is fully offline after the first download. No API keys, no cloud, no telemetry.

### Available voices

| Name | Language | Size |
|---|---|---|
| Lessac | US English | ~63 MB |
| Amy | US English | ~63 MB |
| Ryan | US English, male | ~63 MB |
| Alba | British English | ~63 MB |

### Expected timings

On a typical CPU:

| Step | Time |
|---|---|
| Transcribe 3-second clip | 1 to 2 seconds |
| Generate reply | 3 to 6 seconds |
| Synthesize speech | 1 to 2 seconds |
| **Total per turn** | **6 to 10 seconds** |

A CUDA GPU would be 3 to 4 times faster. The app automatically uses the GPU if `llama-cpp-python` was installed with CUDA support.

## Advanced controls

Per-session settings in the sidebar under **Advanced**:

| Setting | Effect |
|---|---|
| Temperature | 0 is deterministic, higher is more creative |
| Top-p | Nucleus sampling threshold |
| Top-k | Restricts to the k most likely tokens |
| Repeat penalty | Discourages loops |
| Max tokens | Maximum response length |
| Seed | Set to a fixed value for reproducible output |

The **Session** menu at the top right of the Chat tab lets you edit the system prompt and clear messages.

## MCP server

Agent Dojo exposes an MCP server so external AI tools can use your local models as tools.

Supported clients: Claude Desktop, Cursor, Windsurf, and any tool that speaks the Model Context Protocol.

### Available tools

| Tool | Purpose |
|---|---|
| `list_models` | List downloaded models |
| `search_models` | Search Hugging Face for GGUF repositories |
| `list_model_files` | List files in a Hugging Face repository |
| `download_model` | Start a model download |
| `chat` | Send a message to a local model |
| `delete_local_model` | Remove a downloaded model |
| `get_storage_info` | Report disk usage |
| `list_preset_models` | Show recommended starter models |

### Available resources

| URI | Contents |
|---|---|
| `agentdojo://models` | List of downloaded models |
| `agentdojo://storage` | Disk usage of the model library |
| `agentdojo://sessions` | Summary of saved chat sessions |

### Running the server

Test with the MCP Inspector:

```bash
mcp dev mcp_server.py
```

Connect to Claude Desktop by editing `%APPDATA%\Claude\claude_desktop_config.json` on Windows or `~/Library/Application Support/Claude/claude_desktop_config.json` on Mac:

```json
{
  "mcpServers": {
    "agent-dojo": {
      "command": "python",
      "args": ["FULL_PATH_TO_mcp_server.py"],
      "cwd": "FULL_PATH_TO_agent_dojo_folder"
    }
  }
}
```

Replace both paths with the actual locations. Restart Claude Desktop completely (quit from the system tray). You should see a tools icon near the message box.

Then in Claude, type:

> List my local models using Agent Dojo.

Claude calls `list_models` and shows the result.

## Where files live

| Path | Contents |
|---|---|
| `models/` | Downloaded GGUF files and metadata |
| `voices/` | Downloaded Piper voice packs |
| `sessions.json` | Chat history and per-session settings |

All three are in `.gitignore`.

## Project structure

```
agent-dojo/
├── app.py                     Streamlit interface
├── mcp_server.py              MCP server for external clients
├── requirements.txt
├── .gitignore
├── README.md
└── core/
    ├── __init__.py
    ├── downloader.py          Download manager with pause and resume
    ├── eval.py                llama.cpp inference and streaming
    ├── registry.py            Model catalog and search
    ├── sessions.py            Chat session persistence
    └── voice.py               Speech recognition and synthesis
```

## Hardware

- Runs on CPU. 8 GB RAM is enough for a 1.5B model.
- GPU is used automatically if available (CUDA).
- Total disk for all four preset models plus one voice: under 3 GB.

## Known limitations

- Only one model loaded at a time. Switching models in the Chat tab unloads the previous one.
- Voice recording has a 2-second delay between stop and processing. This is a Streamlit limitation, not the app's.
- Browser autoplay policies may require one manual click on the audio player before subsequent replies play automatically.
- MCP server uses stdio transport, one client at a time.

## Troubleshooting

**Model downloads fail with a 401 error**
The repository is private or gated. Add a Hugging Face token in the search field.

**Chat works but voice does not speak**
Browser autoplay was blocked. Click the small play arrow on the audio player once. Subsequent replies will play automatically in the same session.

**Voice transcription is slow**
The first transcription downloads the Whisper model (~75 MB) and initializes it. Subsequent transcriptions are much faster. If it stays slow, your CPU is the bottleneck.

**Piper voices sound robotic**
Try a different voice from the Voices tab. Lessac and Amy are the most natural.

**The app uses too much memory**
Close other apps. Each loaded model uses 0.5 to 1.5 GB of RAM. The voice models add another ~200 MB.

## License

MIT