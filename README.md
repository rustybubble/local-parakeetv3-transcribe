# Video Transcribe

Local, offline transcription for video and audio files using
[NVIDIA Parakeet TDT v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3).
Drop in an MKV, MP4, MP3 (or anything ffmpeg understands), get back text,
SRT subtitles, VTT, and JSON segments with timestamps.

Two ways to use it:

- **Web UI** (`app.py` / `run.bat`) — drag-and-drop in the browser, live
  progress bar with elapsed time and ETA, downloads for every output format.
- **CLI** (`transcribe.py`) — point it at a file or a folder of files.

Everything runs locally. No data leaves the machine.

---

## Requirements

- Python 3.11
- [ffmpeg](https://ffmpeg.org/download.html) on `PATH`
- PyTorch (CPU or CUDA — install separately, see below)

## Setup

1. **Install PyTorch first.** Pick the right wheel for your machine at
   <https://pytorch.org/get-started/locally/>. Examples:

   ```sh
   # CPU only (Windows / Mac / Linux)
   pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu

   # CUDA 12.1 (NVIDIA GPU)
   pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```

2. **Install everything else:**

   ```sh
   pip install -r requirements.txt
   ```

   The first transcription will download the Parakeet model (~600 MB) into
   the HuggingFace cache. Subsequent runs reuse it.

## Usage

### Web UI (recommended)

On Windows: double-click `run.bat`.
On other platforms:

```sh
python -u app.py
```

Then open <http://127.0.0.1:5000>. The model loads in the background
(~30 s) so you can pick a file right away — transcription starts as soon
as the model is ready.

`run.bat` is smart: if the server is already running, double-clicking it
again just opens the browser instead of reloading the model. Leave the
window open between uses to skip the model-load wait.

### Command line

```sh
# Single file (default model, txt + srt output)
python transcribe.py lecture.mkv

# Pick formats
python transcribe.py meeting.mp4 --format srt vtt txt json

# Whole directory, output to one place
python transcribe.py ./recordings --output ./transcripts

# Long-audio mode (recordings > ~15 min) — required on CPU
python transcribe.py long_lecture.mkv --long

# Bigger English-only model
python transcribe.py lecture.mkv --model nvidia/parakeet-tdt-1.1b
```

## Output formats

| Format | What it is                                            |
|--------|-------------------------------------------------------|
| `txt`  | Plain text transcript                                 |
| `srt`  | SubRip subtitles (drop into VLC or any video player)  |
| `vtt`  | WebVTT subtitles                                      |
| `json` | Segments with start/end timestamps                    |

## Notes

- **CPU is supported but slow** — expect roughly 0.2× real time
  (~3-4 min of compute per 18 min of audio). Pass `--long` on the CLI;
  the web UI does this automatically.
- **GPU is much faster** if you have an NVIDIA card and a CUDA-built
  PyTorch.
- Default model auto-detects 25 languages. Use `nvidia/parakeet-tdt-1.1b`
  for English-only with slightly higher accuracy.
- Web uploads are capped at 5 GB.

## Project layout

```
app.py            Flask web server
transcribe.py     Core transcription logic + CLI
templates/
  index.html      Web UI
run.bat           Windows launcher (reuses running server)
requirements.txt
```
