#!/usr/bin/env python3
"""Local MKV/video transcription using faster-whisper (Whisper on CTranslate2)."""

import json
import subprocess
import sys
import tempfile
import argparse
from pathlib import Path

SUPPORTED = {'.mkv', '.mp4', '.avi', '.mov', '.webm', '.m4v',
             '.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac'}

VIDEO_EXTS = {'.mkv', '.mp4', '.avi', '.mov', '.webm', '.m4v'}

# small (244M, multilingual) is the speed/accuracy sweet spot for laptop CPU:
# ~3-5x realtime with INT8, auto-detects language. Override with --model.
DEFAULT_MODEL = "small"


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

def load_model(model_name: str = DEFAULT_MODEL, compute_type: str = "int8"):
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        print("ERROR: faster-whisper not installed.")
        print("  Run: pip install faster-whisper")
        sys.exit(1)

    print(f"Loading model: {model_name}  (first run downloads from HuggingFace)")
    return WhisperModel(model_name, device="cpu", compute_type=compute_type)


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
                print(f"    {label}{seg['text'].strip()}")
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

    faster-whisper returns a generator — iterating is what actually drives
    inference. We collect into a list so the rest of the pipeline can work
    with it.

    ``progress(audio_seconds_done)``: optional callback invoked after each
    segment with how much audio (in seconds) has been processed. The web
    UI uses this for a real progress bar instead of a constant-time guess.

    ``on_segment(seg_dict)``: optional callback invoked with each segment
    as it arrives — used by the CLI's --verbose mode to print live.
    """
    segments_iter, _info = model.transcribe(
        str(wav),
        beam_size=5,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500),
    )
    segs = []
    for s in segments_iter:
        seg = {
            'start': round(s.start, 3),
            'end':   round(s.end, 3),
            'text':  s.text.strip(),
        }
        segs.append(seg)
        if on_segment is not None:
            on_segment(seg)
        if progress is not None:
            progress(s.end)
    return segs


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Transcribe MKV/video files locally with faster-whisper.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Models (multilingual — auto-detect language):
  tiny      ~75 MB    fastest, lowest accuracy
  base      ~140 MB
  small     ~470 MB   good speed/accuracy balance [default]
  medium    ~1.5 GB   slower, more accurate
  large-v3  ~3 GB     most accurate, slow on CPU

English-only (smaller / a bit faster on English speech):
  tiny.en, base.en, small.en, medium.en
  distil-large-v3   ~1.5 GB, distilled large-v3, English

Output formats:
  txt   Plain text transcript
  srt   SubRip subtitles (works in VLC, most video players)
  vtt   WebVTT subtitles
  json  Segments with timestamps as JSON

Examples:
  python transcribe.py lecture.mkv
  python transcribe.py meeting.mkv --format srt vtt txt
  python transcribe.py recordings/ --output ./transcripts
  python transcribe.py lecture.mkv --model medium
  python transcribe.py lecture.mkv --model distil-large-v3
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
                        help='Whisper model name (see Models above)')
    parser.add_argument('--compute-type', default='int8',
                        choices=['int8', 'int8_float32', 'float32'],
                        help='Compute precision (int8 is fastest on CPU; default)')
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
    model = load_model(args.model, args.compute_type)

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
