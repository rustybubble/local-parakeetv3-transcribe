#!/usr/bin/env python3
"""Local MKV/video transcription using NVIDIA Parakeet v3."""

import contextlib
import json
import os
import subprocess
import sys
import tempfile
import argparse
from pathlib import Path

SUPPORTED = {'.mkv', '.mp4', '.avi', '.mov', '.webm', '.m4v',
             '.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac'}

VIDEO_EXTS = {'.mkv', '.mp4', '.avi', '.mov', '.webm', '.m4v'}


def check_ffmpeg():
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
    except FileNotFoundError:
        print("ERROR: ffmpeg not found.")
        print("  Install via: winget install ffmpeg  (or https://ffmpeg.org/download.html)")
        sys.exit(1)


def extract_audio(src: Path, dst: Path):
    """Convert any audio/video file to 16 kHz mono PCM WAV for Parakeet."""
    subprocess.run([
        'ffmpeg', '-y', '-i', str(src),
        '-vn',                  # strip video
        '-ac', '1',             # mono
        '-ar', '16000',         # 16 kHz sample rate required by Parakeet
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

def load_model(model_name: str, long_audio: bool):
    try:
        import nemo.collections.asr as nemo_asr
    except ImportError:
        print("ERROR: NeMo ASR not installed.")
        print("  Run: pip install nemo_toolkit[asr]")
        sys.exit(1)

    print(f"Loading model: {model_name}  (first run downloads ~600 MB)")
    model = nemo_asr.models.ASRModel.from_pretrained(model_name)

    if long_audio:
        # Local attention keeps VRAM usage flat for recordings longer than ~15 min
        print("Long-audio mode: switching to local attention...")
        model.change_attention_model(
            self_attention_model="rel_pos_local_attn",
            att_context_size=[256, 256]
        )

    return model


# ── Core transcription ───────────────────────────────────────────────────────

def transcribe_file(model, src: Path, out_dir: Path, formats: list, verbose: bool):
    print(f"\n[{src.name}]")

    with tempfile.TemporaryDirectory() as tmp:
        if src.suffix.lower() in VIDEO_EXTS or src.suffix.lower() != '.wav':
            wav = Path(tmp) / 'audio.wav'
            print("  Extracting audio...")
            try:
                extract_audio(src, wav)
            except subprocess.CalledProcessError as e:
                raise RuntimeError(f"FFmpeg failed — is the file playable? ({e})")
        else:
            wav = src

        print("  Transcribing...")
        segs = _run_transcription(model, wav)

    if verbose:
        for s in segs:
            label = f"[{s['start']:.1f}s] " if s['start'] else ""
            print(f"    {label}{s['text'].strip()}")

    out_dir.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        out = out_dir / f"{src.stem}.{fmt}"
        WRITERS[fmt](segs, out)
        print(f"  -> {out}")


@contextlib.contextmanager
def _tolerant_tempdir_cleanup():
    """Make tempfile.TemporaryDirectory tolerate cleanup failures on Windows.

    NeMo's transcribe() writes a dataloader manifest.json into a
    TemporaryDirectory that it manages internally. On Windows the file handle
    isn't always released by the time __exit__ runs, so cleanup raises
    WinError 32 — and because that happens *inside* NeMo's `with` block, the
    transcription result is destroyed along with the exception. For a long
    job (e.g. a 90-min file) that means hours of work lost at the very end.

    Patching tempfile.TemporaryDirectory for the duration of the call lets
    NeMo finish and return; any leaked temp dir is small (just the manifest)
    and Windows cleans %TEMP% periodically.
    """
    if os.name != 'nt':
        yield
        return

    original = tempfile.TemporaryDirectory

    class _IgnoreCleanupTempDir(original):
        def __init__(self, *args, **kwargs):
            kwargs.setdefault('ignore_cleanup_errors', True)
            super().__init__(*args, **kwargs)

    tempfile.TemporaryDirectory = _IgnoreCleanupTempDir
    try:
        yield
    finally:
        tempfile.TemporaryDirectory = original


def _run_transcription(model, wav: Path) -> list:
    """Return a list of segment dicts: {start, end, text}."""
    # num_workers=0 avoids spawning dataloader subprocesses, which on Windows
    # can leave manifest.json file handles locked and break repeated runs
    # (the long-lived web server hits this; the one-shot CLI usually doesn't).
    with _tolerant_tempdir_cleanup():
        try:
            output = model.transcribe([str(wav)], timestamps=True, num_workers=0)
            result = output[0]
            text = result.text if hasattr(result, 'text') else str(result)

            if hasattr(result, 'timestamp') and result.timestamp:
                raw = result.timestamp.get('segment', [])
                if raw:
                    return [
                        {
                            'start': round(seg['start'], 3),
                            'end':   round(seg['end'],   3),
                            'text':  seg['segment']
                        }
                        for seg in raw
                    ]
            # Timestamps not available — return single block
            return [{'start': 0.0, 'end': 0.0, 'text': text}]

        except Exception:
            # Fallback: transcribe without timestamps
            output = model.transcribe([str(wav)], num_workers=0)
            text = output[0] if isinstance(output[0], str) else output[0].text
            return [{'start': 0.0, 'end': 0.0, 'text': text}]


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Transcribe MKV/video files locally with NVIDIA Parakeet v3.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Models:
  nvidia/parakeet-tdt-0.6b-v3   600 M params, 25 languages, auto-detects language [default]
  nvidia/parakeet-tdt-1.1b      1.1 B params, English only, slightly more accurate

Output formats:
  txt   Plain text transcript
  srt   SubRip subtitles (works in VLC, most video players)
  vtt   WebVTT subtitles
  json  Segments with timestamps as JSON

Examples:
  python transcribe.py lecture.mkv
  python transcribe.py meeting.mkv --format srt vtt txt
  python transcribe.py recordings/ --output ./transcripts
  python transcribe.py long_lecture.mkv --long
  python transcribe.py lecture.mkv --model nvidia/parakeet-tdt-1.1b
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
    parser.add_argument('--model', '-m', default='nvidia/parakeet-tdt-0.6b-v3',
                        help='Parakeet model to use (see Models above)')
    parser.add_argument('--long', '-l', action='store_true',
                        help='Long-audio mode for recordings over ~15 min (reduces VRAM)')
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
    model = load_model(args.model, args.long)

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
