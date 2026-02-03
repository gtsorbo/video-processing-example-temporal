from __future__ import annotations

import subprocess
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import random
import threading

# In-memory state: how many failures remain for each clip.
# Keyed by absolute clip path string.
_failure_budget: Dict[str, int] = {}
_failure_lock = threading.Lock()


app = FastAPI(title="PseudoAI.biz", version="0.1.0")


class AnalyzeClipRequest(BaseModel):
    clip_uri: str  # file://...


class AnalyzeClipResponse(BaseModel):
    description: str
    keywords: List[str]  # exactly 12
    evidence: Dict


def _path_from_file_uri(uri: str) -> Path:
    if not uri.startswith("file://"):
        raise HTTPException(status_code=400, detail="clip_uri must be file://...")
    return Path(uri[len("file://"):]).expanduser().resolve()

def _get_failure_budget(clip_path: Path) -> int:
    """
    Assign a random failure budget per clip once per server process.
    Budget is number of times we will intentionally fail before succeeding forever.

    Env knobs:
      FAIL_PROB        (default 0.35) probability that a clip will have any failures
      FAIL_MAX         (default 2)    max number of failures for a clip that is chosen to fail
    """
    fail_prob = float(os.getenv("FAIL_PROB", "0.35"))
    fail_max = int(os.getenv("FAIL_MAX", "2"))

    key = str(clip_path)

    with _failure_lock:
        if key not in _failure_budget:
            # Decide whether this clip will fail at all
            if random.random() < fail_prob:
                _failure_budget[key] = random.randint(1, max(1, fail_max))
            else:
                _failure_budget[key] = 0
        return _failure_budget[key]


def _decrement_failure_budget(clip_path: Path) -> int:
    """
    Decrement and return remaining budget after decrement.
    """
    key = str(clip_path)
    with _failure_lock:
        remaining = _failure_budget.get(key, 0)
        if remaining > 0:
            remaining -= 1
            _failure_budget[key] = remaining
        return remaining


def _maybe_fail(clip_path: Path) -> None:
    """
    Random-but-bounded failure simulation. All clips eventually succeed.

    On a failure, randomly choose:
      - 500 (Transient)
      - 429 (Rate limited)
      - Slow response (to trigger client timeout) [optional]

    Env knobs:
      SIMULATE_FAILURES (default true)
      FAIL_PROB         (default 0.35)
      FAIL_MAX          (default 2)
      FAIL_MODE         (default mixed) one of: mixed|500|429|slow
      SLOW_PROB         (default 0.15) only used for mixed
      SLOW_SLEEP_S      (default 10.0)
    """
    if os.getenv("SIMULATE_FAILURES", "true").lower() != "true":
        return

    budget = _get_failure_budget(clip_path)
    if budget <= 0:
        return

    # We will fail this request and decrement budget.
    remaining_after = _decrement_failure_budget(clip_path)

    mode = os.getenv("FAIL_MODE", "mixed").lower()
    slow_prob = float(os.getenv("SLOW_PROB", "0.15"))
    slow_sleep = float(os.getenv("SLOW_SLEEP_S", "10.0"))

    # Choose failure type
    if mode == "500":
        raise HTTPException(status_code=500, detail=f"Simulated transient error (remaining_failures={remaining_after})")
    if mode == "429":
        raise HTTPException(status_code=429, detail=f"Simulated rate limit (remaining_failures={remaining_after})")
    if mode == "slow":
        time.sleep(slow_sleep)
        # If client timeout triggers, you may not even return this, but it's fine.
        raise HTTPException(status_code=504, detail=f"Simulated slow/timeout (remaining_failures={remaining_after})")

    # mixed
    r = random.random()
    if r < slow_prob and os.getenv("SIMULATE_SLOW", "true").lower() == "true":
        time.sleep(slow_sleep)
        raise HTTPException(status_code=504, detail=f"Simulated slow/timeout (remaining_failures={remaining_after})")
    elif r < slow_prob + 0.35:
        raise HTTPException(status_code=429, detail=f"Simulated rate limit (remaining_failures={remaining_after})")
    else:
        raise HTTPException(status_code=500, detail=f"Simulated transient error (remaining_failures={remaining_after})")


def _sample_frames(video_path: Path, max_frames: int) -> List[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise HTTPException(status_code=400, detail="Could not open clip (OpenCV)")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    frames: List[np.ndarray] = []
    if fps > 0 and frame_count > 0:
        idxs = np.linspace(0, max(0, frame_count - 1), num=min(max_frames, frame_count), dtype=int)
        for i in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ok, frame = cap.read()
            if ok and frame is not None:
                frames.append(frame)
    else:
        while len(frames) < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)

    cap.release()
    if not frames:
        raise HTTPException(status_code=400, detail="No frames decoded from clip")
    return frames


def _brightness(frame_bgr: np.ndarray) -> float:
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2].astype(np.float32) / 255.0
    return float(v.mean())


def _colorfulness(frame_bgr: np.ndarray) -> float:
    b, g, r = cv2.split(frame_bgr.astype(np.float32))
    rg = np.abs(r - g)
    yb = np.abs(0.5 * (r + g) - b)
    std_rg, mean_rg = rg.std(), rg.mean()
    std_yb, mean_yb = yb.std(), yb.mean()
    return float(np.sqrt(std_rg**2 + std_yb**2) + 0.3 * np.sqrt(mean_rg**2 + mean_yb**2))


def _motion_score(frames_bgr: List[np.ndarray]) -> float:
    if len(frames_bgr) < 2:
        return 0.0
    mags = []
    prev = cv2.cvtColor(frames_bgr[0], cv2.COLOR_BGR2GRAY)
    prev = cv2.resize(prev, (320, 180), interpolation=cv2.INTER_AREA)
    for f in frames_bgr[1:]:
        nxt = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        nxt = cv2.resize(nxt, (320, 180), interpolation=cv2.INTER_AREA)
        flow = cv2.calcOpticalFlowFarneback(prev, nxt, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        mags.append(float(np.mean(mag)))
        prev = nxt
    return float(np.mean(mags)) if mags else 0.0


# --- Optional: YOLO object detection (realistic) ---
_yolo_model = None

def _load_yolo():
    global _yolo_model
    if _yolo_model is not None:
        return _yolo_model
    try:
        from ultralytics import YOLO
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail="ultralytics not installed (pip install ultralytics) to enable object detection",
        )
    model_name = os.getenv("YOLO_MODEL", "yolov8n.pt")
    _yolo_model = YOLO(model_name)
    return _yolo_model


def _detect_objects(frames: List[np.ndarray]) -> Dict[str, int]:
    if os.getenv("ENABLE_YOLO", "true").lower() != "true":
        return {}

    model = _load_yolo()
    conf = float(os.getenv("YOLO_CONF", "0.35"))
    max_w = int(os.getenv("DETECT_MAX_WIDTH", "640"))

    totals: Dict[str, int] = {}
    for frame in frames:
        h, w = frame.shape[:2]
        if w > max_w:
            scale = max_w / w
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

        results = model.predict(frame, conf=conf, verbose=False)
        r = results[0]
        if r.boxes is None or len(r.boxes) == 0:
            continue

        names = r.names
        for c in r.boxes.cls.tolist():
            label = names.get(int(c), str(int(c)))
            totals[label] = totals.get(label, 0) + 1

    return totals


# --- Optional OCR ---
def _ocr_text(frame_bgr: np.ndarray) -> List[str]:
    if os.getenv("ENABLE_OCR", "false").lower() != "true":
        return []
    try:
        import pytesseract
    except Exception:
        return []
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (0, 0), fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    text = pytesseract.image_to_string(gray) or ""
    lines = []
    for line in text.splitlines():
        line = line.strip()
        if len(line) >= 3 and any(ch.isalnum() for ch in line):
            lines.append(line)
    # unique
    out, seen = [], set()
    for l in lines:
        k = l.lower()
        if k not in seen:
            out.append(l)
            seen.add(k)
    return out[:5]


def _audio_presence_and_silence_ratio(video_path: Path) -> Tuple[bool, float]:
    # Has audio?
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries", "stream=codec_type", "-of", "csv=p=0", str(video_path)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    has_audio = probe.returncode == 0 and probe.stdout.strip() != ""
    if not has_audio:
        return False, 1.0

    # Use silencedetect
    thresh_db = os.getenv("SILENCE_THRESH_DB", "-35dB")
    min_silence = os.getenv("SILENCE_MIN_DUR_S", "0.3")
    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-nostats", "-i", str(video_path),
         "-af", f"silencedetect=noise={thresh_db}:d={min_silence}", "-f", "null", "-"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    out = proc.stderr or ""

    # Duration
    durp = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", str(video_path)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    try:
        duration = float(durp.stdout.strip())
    except Exception:
        duration = 0.0
    if duration <= 0:
        return True, 0.0

    segments = []
    start = None
    for line in out.splitlines():
        line = line.strip()
        if "silence_start:" in line:
            try:
                start = float(line.split("silence_start:")[1].strip())
            except Exception:
                start = None
        if "silence_end:" in line and start is not None:
            try:
                end_part = line.split("silence_end:")[1]
                end_str = end_part.split("|")[0].strip()
                end = float(end_str)
                segments.append((start, end))
            except Exception:
                pass
            start = None

    silent_total = sum(max(0.0, b - a) for a, b in segments)
    ratio = max(0.0, min(1.0, silent_total / duration))
    return True, ratio


def _keywords_from_objects(obj_counts: Dict[str, int]) -> List[str]:
    # Object/person keywords only, exactly 12 unique
    if not obj_counts:
        # honest fallback
        return ["unknown_object"] + [f"object_{i}" for i in range(2, 13)]

    ordered = sorted(obj_counts.items(), key=lambda kv: (-kv[1], kv[0]))
    keywords = [name for name, _ in ordered]

    if "person" in obj_counts and keywords[0] != "person":
        keywords.remove("person")
        keywords.insert(0, "person")

    uniq, seen = [], set()
    for k in keywords:
        kk = k.lower().strip()
        if kk and kk not in seen:
            uniq.append(k)
            seen.add(kk)
        if len(uniq) == 12:
            break

    while len(uniq) < 12:
        uniq.append(f"object_{len(uniq)+1}")

    return uniq


def _description(obj_counts: Dict[str, int], texts: List[str], has_audio: bool, silence_ratio: float,
                 metrics: Dict[str, float]) -> str:
    parts = []

    if obj_counts:
        ordered = sorted(obj_counts.items(), key=lambda kv: (-kv[1], kv[0]))
        top = [name for name, _ in ordered[:6]]
        parts.append(f"Detected objects: {', '.join(top)}.")
        if "person" in obj_counts:
            parts.append("People appear in at least some frames.")
    else:
        parts.append("No confident objects were detected in sampled frames.")

    if texts:
        parts.append(f"On-screen text (sampled): {', '.join([repr(t) for t in texts[:3]])}.")

    if has_audio:
        if silence_ratio >= 0.8:
            parts.append("Audio is present but mostly silent.")
        elif silence_ratio <= 0.2:
            parts.append("Audio is present and mostly non-silent.")
        else:
            parts.append("Audio is present with mixed silence.")
    else:
        parts.append("No audio track detected.")

    # Include a compact “why” to stay grounded but concise
    parts.append(
        f"(brightness={metrics['brightness_mean']:.2f}, motion={metrics['motion_mean']:.2f})"
    )
    return " ".join(parts)


@app.post("/analyze/clip", response_model=AnalyzeClipResponse)
def analyze_clip(req: AnalyzeClipRequest):
    clip_path = _path_from_file_uri(req.clip_uri)
    if not clip_path.exists() or clip_path.stat().st_size == 0:
        raise HTTPException(status_code=400, detail="Clip not found or empty")

    _maybe_fail(clip_path)

    frames = _sample_frames(clip_path, max_frames=int(os.getenv("MAX_FRAMES", "12")))

    brightness_vals = [_brightness(f) for f in frames]
    color_vals = [_colorfulness(f) for f in frames]
    motion = _motion_score(frames)

    metrics = {
        "brightness_mean": float(np.mean(brightness_vals)),
        "colorfulness_mean": float(np.mean(color_vals)),
        "motion_mean": float(motion),
        "frames_sampled": len(frames),
    }

    obj_counts = _detect_objects(frames)

    texts = []
    if os.getenv("ENABLE_OCR", "false").lower() == "true":
        for f in frames:
            for t in _ocr_text(f):
                if t not in texts:
                    texts.append(t)

    has_audio, silence_ratio = _audio_presence_and_silence_ratio(clip_path)

    description = _description(obj_counts, texts, has_audio, silence_ratio, metrics)
    keywords = _keywords_from_objects(obj_counts)

    evidence = {
        "objects": dict(sorted(obj_counts.items(), key=lambda kv: (-kv[1], kv[0]))),
        "text": texts[:5],
        "audio": {"has_audio": has_audio, "silence_ratio_est": round(float(silence_ratio), 3)},
        "metrics": metrics,
    }

    return AnalyzeClipResponse(description=description, keywords=keywords, evidence=evidence)