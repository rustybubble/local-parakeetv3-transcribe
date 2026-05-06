#!/usr/bin/env python3
"""Local MKV/video transcription using whisper.cpp (via pywhispercpp)."""

import json
import subprocess
import sys
import tempfile
import argparse
from pathlib import Path

SUPPORTED = {'.mkv', '.mp4', '.avi', '.mov', '.webm', '.m4v',
             '.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac'}

VIDEO_EXTS = {'.mkv', '.mp4', '.avi', '.mov', '.webm', '.m4v'}

# small-q5_1: ~150 MB, multilingual, q5_1-quantized. On a modern laptop CPU
# this hits ~5-8x realtime — meaningfully faster than the f16 small with
# essentially no accuracy hit. Override with --model on the CLI; see the
# README for the full menu (tiny/base/small/medium/large-v3, plus quantized
# and English-only variants).
DEFAULT_MODEL = "small-q5_1"

# Keep the model cache next to the project so the install is self-contained
# (pywhispercpp's default cache lives in %LOCALAPPDATA% / ~/.local/share,
# which is fine but harder to find or back up).
MODELS_DIR = Path(__file__).parent / "models"


def check_ffmpeg():
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
    except FileNotFoundError:
        print("ERROR: ffmpeg not found.")
        print("  Install via: winget install ffmpeg  (or https://ffmpeg.org/download.html)")
        sys.exit(1)


def extract_audio(src: Path, dst: Path):
    """Convert any audio/video file to 16 kHz mono PCM WAV (Whisper's native format)."""
    subprocess.run([
        'ffmpeg', '-y', '-i', str(src),
        '-vn',                  # strip video
        '-ac', '1',             # mono
        '-ar', '16000',         # 16 kHz sample rate
        '-acodec', 'pcm_s16le', # 16-bit PCM
        str(dst)
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def wav_duration_sec(wav_path: Path) -> float:
    """Duration of a 16 kHz mono 16-bit PCM WAV (the output of extract_audio)."""
    size = wav_path.stat().st_size
    return max(0.0, (size - 44) / 32000.0)  # 44-byte header, 32000 bytes/sec


# ── Timestamp helpers ────────────────────────────────────────────────────────

def _ts(seconds: float, sep: str) -> str:
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"

def ts_srt(s: float) -> str: return _ts(s, ',')
def ts_vtt(s: float) -> str: return _ts(s, '.')


# ── Output writers ───────────────────────────────────────────────────────────

def write_txt(segs: list, path: Path):
    path.write_text('\n'.join(s['text'].strip() for s in segs), encoding='utf-8')


def write_srt(segs: list, path: Path):
    blocks = []
    for i, s in enumerate(segs, 1):
        blocks.append(f"{i}\n{ts_srt(s['start'])} --> {ts_srt(s['end'])}\n{s['text'].strip()}")
    path.write_text('\n\n'.join(blocks) + '\n', encoding='utf-8')


def write_vtt(segs: list, path: Path):
    blocks = ['WEBVTT\n']
    for i, s in enumerate(segs, 1):
        blocks.append(f"{i}\n{ts_vtt(s['start'])} --> {ts_vtt(s['end'])}\n{s['text'].strip()}")
    path.write_text('\n\n'.join(blocks) + '\n', encoding='utf-8')


def write_json(segs: list, path: Path):
    path.write_text(json.dumps({'segments': segs}, indent=2, ensure_ascii=False), encoding='utf-8')


WRITERS = {'txt': write_txt, 'srt': write_srt, 'vtt': write_vtt, 'json': write_json}


# ── Model loading ────────────────────────────────────────────────────────────

def load_model(model_name: str = DEFAULT_MODEL):
    try:
        from pywhispercpp.model import Model
    except ImportError:
        print("ERROR: pywhispercpp not installed.")
        print("  Run: pip install pywhispercpp")
        sys.exit(1)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Loading model: {model_name}  (first run downloads from HuggingFace into ./models)")
    # print_progress / print_realtime would spam stdout from inside whisper.cpp;
    # we get the same info via the streaming segment callback in _run_transcription.
    return Model(
        model_name,
        models_dir=str(MODELS_DIR),
        print_progress=False,
        print_realtime=False,
    )


# ── Core transcription ───────────────────────────────────────────────────────

def transcribe_file(model, src: Path, out_dir: Path, formats: list, verbose: bool):
    print(f"\n[{src.name}]")

    with tempfile.TemporaryDirectory() as tmp:
        if src.suffix.lower() != '.wav':
            wav = Path(tmp) / 'audio.wav'
            print("  Extracting audio...")
            try:
                extract_audio(src, wav)
            except subprocess.CalledProcessError as e:
                raise RuntimeError(f"FFmpeg failed — is the file playable? ({e})")
        else:
            wav = src

        print("  Transcribing...")
        if verbose:
            def _on_seg(seg):
                label = f"[{seg['start']:.1f}s] " if seg['start'] else ""
                print(f"    {label}{seg['text'].strip()}", flush=True)
            segs = _run_transcription(model, wav, on_segment=_on_seg)
        else:
            segs = _run_transcription(model, wav)

    out_dir.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        out = out_dir / f"{src.stem}.{fmt}"
        WRITERS[fmt](segs, out)
        print(f"  -> {out}")


def _run_transcription(model, wav: Path, progress=None, on_segment=None) -> list:
    """Return a list of segment dicts: {start, end, text}.

    whisper.cpp emits segments through a streaming callback as inference
    runs. Each Segment carries t0/t1 in 10ms units (centiseconds) — we
    divide by 100 to get seconds.

    ``progress(audio_seconds_done)`` and ``on_segment(seg_dict)`` are
    optional callbacks the web UI / CLI use for live feedback.
    """
    captured: list = []

    def cb(seg):
        d = {
            'start': round(seg.t0 / 100.0, 3),
            'end':   round(seg.t1 / 100.0, 3),
            'text':  seg.text.strip(),
        }
        captured.append(d)
        if on_segment is not None:
            on_segment(d)
        if progress is not None:
            progress(d['end'])

    model.transcribe(str(wav), new_segment_callback=cb)
    return captured


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Transcribe MKV/video files locally with whisper.cpp.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Models (multilingual unless suffixed .en — auto-detect language):
  tiny / tiny.en        ~75 MB     fastest, lowest accuracy
  base / base.en        ~140 MB
  small / small.en      ~470 MB    good balance
  medium / medium.en    ~1.5 GB    slower, more accurate
  large-v3              ~3 GB      most accurate, slow on CPU
  large-v3-turbo        ~1.6 GB    near-large quality, faster

Quantized variants (recommended on CPU — append to any name above):
  -q5_1   smallest, ~negligible accuracy loss          (default: small-q5_1)
  -q8_0   slightly larger, slightly more accurate
  e.g. small-q5_1, medium-q8_0, large-v3-turbo-q5_0, small.en-q5_1

Output formats:
  txt   Plain text transcript
  srt   SubRip subtitles (works in VLC, most video players)
  vtt   WebVTT subtitles
  json  Segments with timestamps as JSON

Examples:
  python transcribe.py lecture.mkv
  python transcribe.py meeting.mkv --format srt vtt txt
  python transcribe.py recordings/ --output ./transcripts
  python transcribe.py lecture.mkv --model medium-q5_0
  python transcribe.py lecture.mkv --model small.en-q5_1
        """
    )
    parser.add_argument('inputs', nargs='+',
                        help='MKV/video/audio files or a directory')
    parser.add_argument('--output', '-o', type=Path, default=None,
                        help="Output directory (default: a 'transcripts' subfolder next to each input)")
    parser.add_argument('--format', '-f', nargs='+',
                        choices=['txt', 'srt', 'vtt', 'json'], default=['txt', 'srt'],
                        metavar='FMT',
                        help='Output format(s) — txt srt vtt json (default: txt srt)')
    parser.add_argument('--model', '-m', default=DEFAULT_MODEL,
                        help=f'Whisper model name (default: {DEFAULT_MODEL})')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Print each segment as it is produced')
    args = parser.parse_args()

    check_ffmpeg()

    files: list[Path] = []
    for inp in args.inputs:
        p = Path(inp)
        if p.is_dir():
            for ext in SUPPORTED:
                files.extend(sorted(p.glob(f'*{ext}')))
        elif p.exists():
            if p.suffix.lower() in SUPPORTED:
                files.append(p)
            else:
                print(f"Warning: unsupported format '{p.suffix}' — skipping {p.name}")
        else:
            print(f"Warning: file not found — {p}")

    if not files:
        print("No valid input files found.")
        sys.exit(1)

    print(f"\nFiles to transcribe: {len(files)}")
    model = load_model(args.model)

    ok, failed = 0, []
    for f in files:
        out_dir = args.output if args.output else f.parent / "transcripts"
        try:
            transcribe_file(model, f, out_dir, args.format, args.verbose)
            ok += 1
        except Exception as e:
            print(f"  ERROR: {e}")
            failed.append(f.name)

    print(f"\nDone: {ok}/{len(files)} succeeded.")
    if failed:
        print("Failed:", ', '.join(failed))


if __name__ == '__main__':
    main()
