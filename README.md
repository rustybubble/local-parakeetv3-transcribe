# Video Transcribe

Local, offline transcription for video and audio files using
[faster-whisper](https://github.com/SYSTRAN/faster-whisper) — Whisper running
on CTranslate2 with INT8 quantization, which is roughly an order of magnitude
faster than vanilla Whisper on CPU. Drop in an MKV, MP4, MP3 (or anything
ffmpeg understands), get back text, SRT subtitles, VTT, and JSON segments
with timestamps.

Two ways to use it:

- **Web UI** (`app.py` / `run.bat`) — drag-and-drop in the browser, live
  progress bar that tracks real audio progress, downloads for every output
  format.
- **CLI** (`transcribe.py`) — point it at a file or a folder of files.

Everything runs locally. No data leaves the machine.

---

## Requirements

- Python 3.10+ (3.11 recommended)
- [ffmpeg](https://ffmpeg.org/download.html) on `PATH`

That's it — no separate PyTorch install. CTranslate2 ships prebuilt CPU
wheels, so `pip install` does the right thing on Windows / macOS / Linux.

## Setup

```sh
pip install -r requirements.txt
```

The first transcription will download the Whisper model into the HuggingFace
cache (`small` is ~470 MB; INT8-quantized weights are smaller). Subsequent
runs reuse it.

## Usage

### Web UI (recommended)

On Windows: double-click `run.bat`.
On other platforms:

```sh
python -u app.py
```

Then open <http://127.0.0.1:5000>. The model loads in the background
(~10 s) so you can pick a file right away — transcription starts as soon
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

# Bigger / more accurate model
python transcribe.py lecture.mkv --model medium

# Distilled large (English only, near-large quality at small-ish speed)
python transcribe.py lecture.mkv --model distil-large-v3

# Print each segment as it's produced
python transcribe.py lecture.mkv --verbose
```

## Models

| Name              | Size     | Notes                                          |
|-------------------|----------|------------------------------------------------|
| `tiny`            | ~75 MB   | Fastest, lowest accuracy                       |
| `base`            | ~140 MB  |                                                |
| `small` (default) | ~470 MB  | Good speed/accuracy balance, multilingual      |
| `medium`          | ~1.5 GB  | Slower, more accurate                          |
| `large-v3`        | ~3 GB    | Most accurate, slow on CPU                     |
| `distil-large-v3` | ~1.5 GB  | Distilled large, English-only, fast            |

Append `.en` to `tiny`/`base`/`small`/`medium` for English-only variants
(slightly faster and a bit more accurate on English speech).

## Output formats

| Format | What it is                                            |
|--------|-------------------------------------------------------|
| `txt`  | Plain text transcript                                 |
| `srt`  | SubRip subtitles (drop into VLC or any video player)  |
| `vtt`  | WebVTT subtitles                                      |
| `json` | Segments with start/end timestamps                    |

## Notes

- **CPU speed (rough rule of thumb on a modern laptop):**
  - `small` INT8 — ~3-5× realtime (90 min file in ~20-30 min)
  - `medium` INT8 — ~1-2× realtime
  - `tiny` / `base` INT8 — ~10× realtime if you don't mind the accuracy hit
- VAD filtering is on by default to skip silences and reduce Whisper's
  occasional hallucinations on non-speech audio.
- The web progress bar shows **real** audio progress: the ETA refines itself
  from observed throughput once ~30 s of audio has been processed.
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
