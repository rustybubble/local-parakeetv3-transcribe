# Video Transcribe

Local, offline transcription for video and audio files using
[whisper.cpp](https://github.com/ggerganov/whisper.cpp) — Whisper compiled
to native code with quantized models, the fastest CPU Whisper backend.
Drop in an MKV, MP4, MP3 (or anything ffmpeg understands), get back text,
SRT subtitles, VTT, and JSON segments with timestamps.

Two ways to use it:

- **Web UI** (`run.bat`) — drag-and-drop in the browser, live progress bar
  that tracks real audio progress, downloads for every output format.
- **CLI** (`transcribe.py`) — point it at a file or a folder of files.

Everything runs locally. No data leaves the machine.

---

## Setup (Windows — easiest path)

1. **Install Python 3.11** from <https://www.python.org/downloads/>.
   Tick **"Add Python to PATH"** in the installer.
2. **Install ffmpeg** in PowerShell or Command Prompt:
   ```
   winget install ffmpeg
   ```
   Close and reopen the terminal afterwards so `ffmpeg` is on PATH.
3. **Double-click `setup.bat`** in this folder.
   It creates a `.venv\` and installs everything. Wait for "Setup complete".
4. **Double-click `run.bat`**.
   Your browser opens to <http://127.0.0.1:5000>. The first time, the app
   downloads the whisper.cpp model (~150 MB) into `.\models\`. From then on
   it's instant.

That's it. Drop a file in, click Transcribe, get your transcript.

## Setup (macOS / Linux)

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
brew install ffmpeg          # macOS
# or:  sudo apt install ffmpeg

python -u app.py
```

Then open <http://127.0.0.1:5000>.

## Using the web UI

The first transcription kicks off as soon as the model is loaded (~1-2 s
after the download finishes on first run, instant after that). The progress
bar shows **real audio progress** — it self-corrects from observed
throughput so the ETA is accurate.

Outputs land in the `transcripts\` folder next to the project, and there are
download links for every format (txt, srt, vtt, json) right in the UI.

`run.bat` is smart: if the server is already running, double-clicking it
again just opens the browser instead of reloading the model. Leave the
window open between uses to skip the model load entirely.

## Using the command line

```sh
# Single file (default model, txt + srt output)
python transcribe.py lecture.mkv

# Pick output formats
python transcribe.py meeting.mp4 --format srt vtt txt json

# Whole directory, all output to one place
python transcribe.py ./recordings --output ./transcripts

# Bigger / more accurate model (still fast on CPU thanks to quantization)
python transcribe.py lecture.mkv --model medium-q5_0

# English-only model (slightly faster, slightly more accurate on English)
python transcribe.py lecture.mkv --model small.en-q5_1

# Print each segment as it is produced
python transcribe.py lecture.mkv --verbose
```

## Models

The default is **`small-q5_1`** — multilingual, ~150 MB, very strong
speed/accuracy tradeoff on a laptop CPU.

| Name             | Size     | Notes                                    |
|------------------|----------|------------------------------------------|
| `tiny-q5_1`      | ~30 MB   | Fastest, lowest accuracy                 |
| `base-q5_1`      | ~60 MB   |                                          |
| `small-q5_1` ★   | ~150 MB  | **Default.** Good balance, multilingual  |
| `medium-q5_0`    | ~500 MB  | Slower, more accurate                    |
| `large-v3-q5_0`  | ~1 GB    | Most accurate, slow on CPU               |
| `large-v3-turbo-q5_0` | ~600 MB | Near-large quality, faster          |

Append `.en` to `tiny`/`base`/`small`/`medium` for English-only variants
(slightly faster and a bit more accurate on English speech). The `-q8_0`
variant of any model is slightly larger but slightly more accurate than
`-q5_1`. Drop the suffix entirely for the f16 reference (largest, slowest
on CPU — no reason to use these unless you're chasing accuracy).

Models auto-download on first use into `.\models\`, so to switch model
you just pass `--model <name>` (CLI) or edit `DEFAULT_MODEL` in
`transcribe.py` (web UI).

## Output formats

| Format | What it is                                            |
|--------|-------------------------------------------------------|
| `txt`  | Plain text transcript                                 |
| `srt`  | SubRip subtitles (drop into VLC or any video player)  |
| `vtt`  | WebVTT subtitles                                      |
| `json` | Segments with start/end timestamps                    |

## Notes

- **CPU speed (rough rule of thumb on a modern laptop):**
  - `small-q5_1` — ~5-8× realtime (90 min file in ~12-18 min)
  - `medium-q5_0` — ~2-3× realtime
  - `tiny-q5_1` / `base-q5_1` — ~10-15× realtime if you don't mind the accuracy hit
- Web uploads are capped at 5 GB.

## Project layout

```
app.py            Flask web server
transcribe.py     Core transcription logic + CLI
templates/
  index.html      Web UI
setup.bat         One-time Windows setup (creates .venv\)
run.bat           Windows launcher (reuses running server)
requirements.txt
.venv\            Python virtual environment (created by setup.bat)
models\           Downloaded whisper.cpp models (created on first run)
transcripts\      Output transcripts
```
