"""Local web UI for Parakeet v3 transcription.

Run with:  python app.py
Then open: http://127.0.0.1:5000
"""

import os
import sys
import time
import threading
import tempfile
import uuid
import webbrowser
from pathlib import Path

from flask import Flask, abort, jsonify, render_template, request, send_file

from transcribe import (
    _run_transcription,
    extract_audio,
    load_model,
    wav_duration_sec,
    write_json,
    write_srt,
    write_txt,
    write_vtt,
)

ROOT = Path(__file__).parent
TRANSCRIPTS = ROOT / "transcripts"
TRANSCRIPTS.mkdir(exist_ok=True)

ALLOWED = {
    ".mkv", ".mp4", ".mov", ".avi", ".webm", ".m4v",
    ".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac",
}

# CPU transcription on this box runs at ~0.2x real time (memory: 18 min in 3:30).
# Use 0.25 as a slightly conservative estimate so the bar lands a bit early
# rather than overshooting and stalling at 99%.
CPU_TRANSCRIBE_RATIO = 0.25
MIN_ESTIMATE_SEC = 15.0

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024 * 1024  # 5 GB

_model = None
_model_lock = threading.Lock()
_model_ready = threading.Event()
_model_load_started = time.time()
_job_lock = threading.Lock()  # serialize transcriptions; the model isn't reentrant
_jobs: dict[str, dict] = {}


def get_model():
    global _model
    with _model_lock:
        if _model is None:
            print("Loading Parakeet v3 (long-audio mode)...", flush=True)
            _model = load_model("nvidia/parakeet-tdt-0.6b-v3", long_audio=True)
            _model_ready.set()
            print(f"Model loaded in {time.time() - _model_load_started:.1f}s.", flush=True)
    return _model


def _set_stage(job: dict, status: str, progress: str, estimated_total_sec: float | None = None):
    job["status"] = status
    job["progress"] = progress
    job["stage_started"] = time.time()
    if estimated_total_sec is not None:
        job["estimated_total_sec"] = round(estimated_total_sec, 1)
    else:
        job.pop("estimated_total_sec", None)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/ready")
def ready():
    return jsonify({
        "ready": _model_ready.is_set(),
        "elapsed_sec": round(time.time() - _model_load_started, 1),
    })


@app.route("/transcribe", methods=["POST"])
def transcribe():
    f = request.files.get("file")
    if not f or not f.filename:
        return jsonify({"error": "No file uploaded"}), 400

    ext = Path(f.filename).suffix.lower()
    if ext not in ALLOWED:
        return jsonify({"error": f"Unsupported format: {ext}"}), 400

    job_id = uuid.uuid4().hex[:8]
    fd, tmp_path = tempfile.mkstemp(suffix=ext, prefix=f"transcribe_{job_id}_")
    os.close(fd)
    src = Path(tmp_path)
    f.save(src)

    job = {"filename": f.filename}
    _set_stage(job, "queued", "Queued...")
    _jobs[job_id] = job
    threading.Thread(target=_run_job, args=(job_id, src), daemon=True).start()
    return jsonify({"job_id": job_id})


def _run_job(job_id: str, src: Path):
    job = _jobs[job_id]
    try:
        with _job_lock:
            if not _model_ready.is_set():
                _set_stage(job, "loading_model",
                           "Loading model (one-time, ~30s on first start)...")
            model = get_model()  # blocks until the background loader finishes

            with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
                wav = Path(tmp) / "audio.wav"
                _set_stage(job, "extracting", "Extracting audio with ffmpeg...")
                print(f"[{job_id}] Extracting audio from {job['filename']}...", flush=True)
                extract_audio(src, wav)

                audio_dur = wav_duration_sec(wav)
                job["audio_duration_sec"] = round(audio_dur, 1)
                estimate = max(MIN_ESTIMATE_SEC, audio_dur * CPU_TRANSCRIBE_RATIO)

                mins = audio_dur / 60
                _set_stage(
                    job, "transcribing",
                    f"Transcribing {mins:.1f} min of audio with Parakeet v3...",
                    estimated_total_sec=estimate,
                )
                print(f"[{job_id}] Transcribing {mins:.1f} min "
                      f"(estimated ~{estimate:.0f}s on CPU)...", flush=True)
                segs = _run_transcription(model, wav)

            _set_stage(job, "writing", "Writing transcript files...")
            stem = Path(job["filename"]).stem or f"transcript_{job_id}"
            write_txt(segs, TRANSCRIPTS / f"{stem}.txt")
            write_srt(segs, TRANSCRIPTS / f"{stem}.srt")
            write_vtt(segs, TRANSCRIPTS / f"{stem}.vtt")
            write_json(segs, TRANSCRIPTS / f"{stem}.json")
            job["stem"] = stem

            _set_stage(job, "done", "Complete")
            job["text"] = "\n".join(s["text"].strip() for s in segs)
            job["segment_count"] = len(segs)
            print(f"[{job_id}] Done — {len(segs)} segments.", flush=True)
    except Exception as e:
        job["status"] = "error"
        job["error"] = str(e)
        print(f"[{job_id}] ERROR: {e}", flush=True)
    finally:
        try:
            src.unlink()
        except OSError:
            pass


@app.route("/status/<job_id>")
def status(job_id):
    job = _jobs.get(job_id)
    if not job:
        return jsonify({"status": "unknown"}), 404
    payload = dict(job)
    if "stage_started" in job:
        payload["elapsed_sec"] = round(time.time() - job["stage_started"], 1)
    return jsonify(payload)


@app.route("/download/<job_id>/<fmt>")
def download(job_id, fmt):
    if fmt not in ("txt", "srt", "vtt", "json"):
        abort(400)
    job = _jobs.get(job_id)
    if not job or "stem" not in job:
        abort(404)
    path = TRANSCRIPTS / f"{job['stem']}.{fmt}"
    if not path.exists():
        abort(404)
    return send_file(path, as_attachment=True, download_name=f"{job['stem']}.{fmt}")


if __name__ == "__main__":
    url = "http://127.0.0.1:5000"
    threading.Thread(target=get_model, daemon=True, name="model-loader").start()
    print(f"Server up at {url} — model is loading in the background (~30s).", flush=True)
    print("You can pick a file in the browser right away; the first transcription", flush=True)
    print("will start as soon as the model finishes loading.\n", flush=True)
    threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
