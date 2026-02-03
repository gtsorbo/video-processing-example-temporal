from __future__ import annotations

import os
import random
import re
import subprocess
import threading
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# ----------------------------
# Failure simulation (unchanged)
# ----------------------------
_failure_budget: Dict[str, int] = {}
_failure_lock = threading.Lock()

app = FastAPI(title="PseudoAI.biz", version="0.2.0")


class AnalyzeClipRequest(BaseModel):
    clip_uri: str  # file://...


class AnalyzeClipResponse(BaseModel):
    description: str
    keywords: List[str]  # exactly 12
    evidence: Dict


def _path_from_file_uri(uri: str) -> Path:
    if not uri.startswith("file://"):
        raise HTTPException(status_code=400, detail="clip_uri must be file://...")
    return Path(uri[len("file://") :]).expanduser().resolve()


def _get_failure_budget(clip_path: Path) -> int:
    fail_prob = float(os.getenv("FAIL_PROB", "0.35"))
    fail_max = int(os.getenv("FAIL_MAX", "2"))
    key = str(clip_path)
    with _failure_lock:
        if key not in _failure_budget:
            if random.random() < fail_prob:
                _failure_budget[key] = random.randint(1, max(1, fail_max))
            else:
                _failure_budget[key] = 0
        return _failure_budget[key]


def _decrement_failure_budget(clip_path: Path) -> int:
    key = str(clip_path)
    with _failure_lock:
        remaining = _failure_budget.get(key, 0)
        if remaining > 0:
            remaining -= 1
            _failure_budget[key] = remaining
        return remaining


def _maybe_fail(clip_path: Path) -> None:
    if os.getenv("SIMULATE_FAILURES", "true").lower() != "true":
        return

    budget = _get_failure_budget(clip_path)
    if budget <= 0:
        return

    remaining_after = _decrement_failure_budget(clip_path)

    mode = os.getenv("FAIL_MODE", "mixed").lower()
    slow_prob = float(os.getenv("SLOW_PROB", "0.15"))
    slow_sleep = float(os.getenv("SLOW_SLEEP_S", "10.0"))

    if mode == "500":
        raise HTTPException(status_code=500, detail=f"Simulated transient error (remaining_failures={remaining_after})")
    if mode == "429":
        raise HTTPException(status_code=429, detail=f"Simulated rate limit (remaining_failures={remaining_after})")
    if mode == "slow":
        time.sleep(slow_sleep)
        raise HTTPException(status_code=504, detail=f"Simulated slow/timeout (remaining_failures={remaining_after})")

    r = random.random()
    if r < slow_prob and os.getenv("SIMULATE_SLOW", "true").lower() == "true":
        time.sleep(slow_sleep)
        raise HTTPException(status_code=504, detail=f"Simulated slow/timeout (remaining_failures={remaining_after})")
    elif r < slow_prob + 0.35:
        raise HTTPException(status_code=429, detail=f"Simulated rate limit (remaining_failures={remaining_after})")
    else:
        raise HTTPException(status_code=500, detail=f"Simulated transient error (remaining_failures={remaining_after})")


# ----------------------------
# Smarter frame sampling
# ----------------------------

@dataclass
class SampledFrames:
    frames_bgr: List[np.ndarray]
    frame_indexes: List[int]
    fps: float
    frame_count: int


def _read_frame(cap: cv2.VideoCapture, idx: int) -> Optional[np.ndarray]:
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
    ok, frame = cap.read()
    if not ok or frame is None:
        return None
    return frame


def _edge_density(gray_small: np.ndarray) -> float:
    # Fraction of edge pixels
    edges = cv2.Canny(gray_small, 80, 160)
    return float((edges > 0).mean())


def _frame_interest(prev_gray_small: Optional[np.ndarray], gray_small: np.ndarray) -> float:
    # "Interestingness" = motion-ish change + edge density
    ed = _edge_density(gray_small)
    if prev_gray_small is None:
        return ed
    diff = cv2.absdiff(prev_gray_small, gray_small)
    diff_score = float(diff.mean()) / 255.0
    return 0.65 * diff_score + 0.35 * ed


def _sample_frames_smart(video_path: Path, max_frames: int) -> SampledFrames:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise HTTPException(status_code=400, detail="Could not open clip (OpenCV)")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    # Candidate pool (larger than max_frames), uniformly across the clip
    candidate_count = int(os.getenv("CANDIDATE_FRAMES", "30"))
    candidate_count = max(candidate_count, max_frames)

    if fps <= 0 or frame_count <= 0:
        # Fallback: sequential read up to candidate_count
        candidates: List[Tuple[int, np.ndarray]] = []
        idx = 0
        while len(candidates) < candidate_count:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            candidates.append((idx, frame))
            idx += 1
    else:
        idxs = np.linspace(0, max(0, frame_count - 1), num=min(candidate_count, frame_count), dtype=int)
        candidates = []
        for i in idxs:
            frame = _read_frame(cap, int(i))
            if frame is not None:
                candidates.append((int(i), frame))

    cap.release()

    if not candidates:
        raise HTTPException(status_code=400, detail="No frames decoded from clip")

    # Score candidates and pick:
    # - Always include first/last if available
    # - Include top-K "interesting" frames
    scored: List[Tuple[float, int, np.ndarray]] = []
    prev_small = None
    for idx, frame in candidates:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (320, 180), interpolation=cv2.INTER_AREA)
        score = _frame_interest(prev_small, small)
        prev_small = small
        scored.append((score, idx, frame))

    # Force include first/last
    must_idxs = set()
    must_idxs.add(candidates[0][0])
    must_idxs.add(candidates[-1][0])

    scored_sorted = sorted(scored, key=lambda t: (-t[0], t[1]))
    picked: Dict[int, np.ndarray] = {}

    # Add must frames first
    for idx, frame in candidates:
        if idx in must_idxs:
            picked[idx] = frame

    # Add top interesting frames
    for _, idx, frame in scored_sorted:
        if len(picked) >= max_frames:
            break
        picked.setdefault(idx, frame)

    # If still short, fill by uniform order
    if len(picked) < max_frames:
        for idx, frame in candidates:
            if len(picked) >= max_frames:
                break
            picked.setdefault(idx, frame)

    frame_indexes = sorted(picked.keys())
    frames = [picked[i] for i in frame_indexes]
    return SampledFrames(frames_bgr=frames, frame_indexes=frame_indexes, fps=fps, frame_count=frame_count)


# ----------------------------
# Visual metrics
# ----------------------------

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


def _scene_tags(metrics: Dict[str, float], audio: Dict) -> List[str]:
    tags = []

    b = metrics.get("brightness_mean", 0.5)
    c = metrics.get("colorfulness_mean", 0.0)
    m = metrics.get("motion_mean", 0.0)

    # brightness buckets
    if b >= 0.70:
        tags.append("bright")
    elif b <= 0.35:
        tags.append("dim")
    else:
        tags.append("normal_light")

    # colorfulness buckets (these numbers are heuristic for Hasler/Süsstrunk style metric)
    if c >= 45:
        tags.append("vibrant")
    elif c <= 20:
        tags.append("muted")
    else:
        tags.append("moderate_color")

    # motion buckets
    if m >= 1.2:
        tags.append("fast_motion")
    elif m <= 0.35:
        tags.append("static")
    else:
        tags.append("moderate_motion")

    # audio buckets
    if not audio.get("has_audio", False):
        tags.append("no_audio")
    else:
        sr = float(audio.get("silence_ratio_est", 0.0))
        if sr >= 0.8:
            tags.append("mostly_silent")
        elif sr <= 0.2:
            tags.append("mostly_sound")
        else:
            tags.append("mixed_sound")

        # optional: volume bucket if available
        mean_db = audio.get("mean_volume_db")
        if isinstance(mean_db, (int, float)):
            if mean_db > -18:
                tags.append("loud")
            elif mean_db < -35:
                tags.append("quiet")
            else:
                tags.append("medium_volume")

    # keep it short
    uniq, seen = [], set()
    for t in tags:
        if t not in seen:
            uniq.append(t)
            seen.add(t)
    return uniq[:6]


# ----------------------------
# YOLO object detection (improved: presence + conf)
# ----------------------------

_yolo_model = None


def _load_yolo():
    global _yolo_model
    if _yolo_model is not None:
        return _yolo_model
    try:
        from ultralytics import YOLO
    except Exception:
        raise HTTPException(
            status_code=500,
            detail="ultralytics not installed (pip install ultralytics) to enable object detection",
        )
    model_name = os.getenv("YOLO_MODEL", "yolov8n.pt")
    _yolo_model = YOLO(model_name)
    return _yolo_model


@dataclass
class ObjectStats:
    total_counts: Dict[str, int]
    frame_presence: Dict[str, int]  # in how many sampled frames did label appear?
    avg_conf: Dict[str, float]


def _detect_objects(frames: List[np.ndarray]) -> ObjectStats:
    if os.getenv("ENABLE_YOLO", "true").lower() != "true":
        return ObjectStats(total_counts={}, frame_presence={}, avg_conf={})

    model = _load_yolo()
    conf_th = float(os.getenv("YOLO_CONF", "0.35"))
    max_w = int(os.getenv("DETECT_MAX_WIDTH", "640"))

    total = Counter()
    presence = Counter()
    conf_sum = defaultdict(float)
    conf_n = Counter()

    for frame in frames:
        h, w = frame.shape[:2]
        img = frame
        if w > max_w:
            scale = max_w / w
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

        results = model.predict(img, conf=conf_th, verbose=False)
        r = results[0]
        if r.boxes is None or len(r.boxes) == 0:
            continue

        names = r.names
        frame_labels = set()

        # ultralytics: r.boxes.cls and r.boxes.conf are tensors
        cls_list = r.boxes.cls.tolist()
        conf_list = r.boxes.conf.tolist() if getattr(r.boxes, "conf", None) is not None else [1.0] * len(cls_list)

        for c, cf in zip(cls_list, conf_list):
            label = names.get(int(c), str(int(c)))
            total[label] += 1
            conf_sum[label] += float(cf)
            conf_n[label] += 1
            frame_labels.add(label)

        for label in frame_labels:
            presence[label] += 1

    avg_conf = {k: (conf_sum[k] / max(1, conf_n[k])) for k in conf_sum.keys()}
    return ObjectStats(
        total_counts=dict(total),
        frame_presence=dict(presence),
        avg_conf=avg_conf,
    )


# ----------------------------
# OCR (improved: tokenization)
# ----------------------------

_STOPWORDS = {
    "the", "and", "that", "this", "with", "from", "your", "you", "for", "are", "was", "were", "have",
    "has", "not", "but", "all", "any", "can", "our", "out", "get", "new", "now", "off", "sale",
    "in", "on", "at", "to", "of", "a", "an", "it", "is", "as", "by", "or"
}


def _ocr_lines(frame_bgr: np.ndarray) -> List[str]:
    if os.getenv("ENABLE_OCR", "false").lower() != "true":
        return []
    try:
        import pytesseract
    except Exception:
        return []

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (0, 0), fx=1.6, fy=1.6, interpolation=cv2.INTER_CUBIC)
    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    text = pytesseract.image_to_string(gray) or ""

    lines = []
    for line in text.splitlines():
        line = line.strip()
        if len(line) >= 3 and any(ch.isalnum() for ch in line):
            lines.append(line)

    # unique, preserve order
    out, seen = [], set()
    for l in lines:
        k = l.lower()
        if k not in seen:
            out.append(l)
            seen.add(k)
    return out[:8]


def _ocr_tokens(lines: List[str], max_tokens: int = 4) -> List[str]:
    tokens: List[str] = []
    for line in lines:
        s = line.lower()
        s = re.sub(r"[^a-z0-9\s]", " ", s)
        for w in s.split():
            if len(w) < 4:
                continue
            if w in _STOPWORDS:
                continue
            tokens.append(w)

    # frequency-based pick
    c = Counter(tokens)
    picked = [w for w, _ in c.most_common(max_tokens)]
    # unique preserve
    out, seen = [], set()
    for w in picked:
        if w not in seen:
            out.append(w)
            seen.add(w)
    return out[:max_tokens]


# ----------------------------
# Audio (improved: silence ratio + mean volume)
# ----------------------------

def _audio_presence_and_silence_ratio(video_path: Path) -> Tuple[bool, float]:
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries", "stream=codec_type", "-of", "csv=p=0", str(video_path)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    has_audio = probe.returncode == 0 and probe.stdout.strip() != ""
    if not has_audio:
        return False, 1.0

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


_VOL_RE = re.compile(r"mean_volume:\s*(-?\d+(?:\.\d+)?)\s*dB")


def _mean_volume_db(video_path: Path) -> Optional[float]:
    # Fast-ish: analyze whole file; you can cap duration by adding -t N if needed.
    # volumedetect is coarse but useful.
    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-nostats", "-i", str(video_path), "-af", "volumedetect", "-f", "null", "-"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    text = proc.stderr or ""
    m = _VOL_RE.search(text)
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


def _audio_features(video_path: Path) -> Dict:
    has_audio, silence_ratio = _audio_presence_and_silence_ratio(video_path)
    mean_db = _mean_volume_db(video_path) if has_audio and os.getenv("ENABLE_VOLUME", "true").lower() == "true" else None
    out = {
        "has_audio": has_audio,
        "silence_ratio_est": round(float(silence_ratio), 3),
    }
    if mean_db is not None:
        out["mean_volume_db"] = round(float(mean_db), 2)
    return out


# ----------------------------
# Keywords + description (improved)
# ----------------------------

def _object_keywords(stats: ObjectStats, max_items: int = 6) -> List[str]:
    """
    Score objects by (frame_presence, then total_counts) with a light confidence bump.
    """
    items = []
    for label, total in stats.total_counts.items():
        pres = stats.frame_presence.get(label, 0)
        conf = stats.avg_conf.get(label, 0.0)
        score = pres * 2.0 + total * 0.5 + conf * 0.5
        items.append((score, pres, total, label))

    items.sort(key=lambda t: (-t[0], -t[1], -t[2], t[3]))
    labels = [t[3] for t in items[:max_items]]

    # Person first if present at all
    if "person" in stats.total_counts and "person" in labels and labels[0] != "person":
        labels.remove("person")
        labels.insert(0, "person")

    return labels


def _compose_keywords(stats: ObjectStats, scene_tags: List[str], ocr_tokens: List[str]) -> List[str]:
    """
    Always return exactly 12 keywords:
      - up to 5 objects
      - up to 4 scene/audio tags
      - up to 3 OCR tokens
    De-dup and backfill with sensible generic tags (no object_7 spam).
    """
    obj = _object_keywords(stats, max_items=5)
    tags = scene_tags[:4]
    ocr = ocr_tokens[:3]

    candidates = obj + tags + ocr

    # de-dup (case-insensitive), keep order
    out, seen = [], set()
    for k in candidates:
        kk = k.strip().lower()
        if not kk or kk in seen:
            continue
        out.append(k)
        seen.add(kk)
        if len(out) >= 12:
            break

    # backfill with generic-but-honest tags
    backfill_pool = ["video", "scene", "analysis", "no_clear_subject", "visual_content", "clip"]
    for k in backfill_pool:
        if len(out) >= 12:
            break
        if k not in seen:
            out.append(k)
            seen.add(k)

    # If still short (rare), pad with stable placeholders
    while len(out) < 12:
        out.append(f"tag_{len(out)+1}")

    return out[:12]


def _description(stats: ObjectStats, ocr_lines: List[str], audio: Dict, metrics: Dict[str, float]) -> str:
    """
    Human-ish template:
      - Primary subject(s)
      - Motion/scene feel
      - Audio
      - On-screen text (only if useful)
    """
    parts = []

    top_objs = _object_keywords(stats, max_items=4)
    if top_objs:
        if len(top_objs) == 1:
            parts.append(f"The clip appears to contain {top_objs[0]}.")
        else:
            parts.append(f"The clip appears to contain {', '.join(top_objs[:-1])}, and {top_objs[-1]}.")
        if "person" in stats.frame_presence:
            pres = stats.frame_presence.get("person", 0)
            parts.append(f"A person is visible in about {pres} of the sampled frames.")
    else:
        parts.append("No confident objects were detected in the sampled frames.")

    m = metrics.get("motion_mean", 0.0)
    if m >= 1.2:
        parts.append("There is substantial motion across the sampled frames.")
    elif m <= 0.35:
        parts.append("The scene looks mostly static with little motion.")
    else:
        parts.append("There is moderate motion across the sampled frames.")

    if audio.get("has_audio", False):
        sr = float(audio.get("silence_ratio_est", 0.0))
        if sr >= 0.8:
            parts.append("Audio is present but mostly silent.")
        elif sr <= 0.2:
            parts.append("Audio is present and mostly non-silent.")
        else:
            parts.append("Audio is present with mixed silence.")
        if "mean_volume_db" in audio:
            parts.append(f"(mean volume ~{audio['mean_volume_db']} dB)")
    else:
        parts.append("No audio track was detected.")

    if ocr_lines:
        toks = _ocr_tokens(ocr_lines, max_tokens=3)
        if toks:
            parts.append(f"On-screen text may include: {', '.join(toks)}.")

    # Keep metric “why” out of the main prose; clients can use evidence for that.
    return " ".join(parts).strip()


# ----------------------------
# Endpoint
# ----------------------------

@app.post("/analyze/clip", response_model=AnalyzeClipResponse)
def analyze_clip(req: AnalyzeClipRequest):
    clip_path = _path_from_file_uri(req.clip_uri)
    if not clip_path.exists() or clip_path.stat().st_size == 0:
        raise HTTPException(status_code=400, detail="Clip not found or empty")

    _maybe_fail(clip_path)

    sampled = _sample_frames_smart(clip_path, max_frames=int(os.getenv("MAX_FRAMES", "12")))
    frames = sampled.frames_bgr

    brightness_vals = [_brightness(f) for f in frames]
    color_vals = [_colorfulness(f) for f in frames]
    motion = _motion_score(frames)

    metrics = {
        "brightness_mean": float(np.mean(brightness_vals)),
        "colorfulness_mean": float(np.mean(color_vals)),
        "motion_mean": float(motion),
        "frames_sampled": len(frames),
        "frame_indexes": sampled.frame_indexes,
        "fps": sampled.fps,
        "frame_count": sampled.frame_count,
    }

    obj_stats = _detect_objects(frames)

    # OCR: run only on a few frames most likely to have readable text.
    ocr_lines: List[str] = []
    if os.getenv("ENABLE_OCR", "false").lower() == "true":
        # pick top 4 frames by edge density (cheap heuristic for text/structure)
        scored = []
        for idx, f in zip(sampled.frame_indexes, frames):
            gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
            small = cv2.resize(gray, (320, 180), interpolation=cv2.INTER_AREA)
            scored.append((_edge_density(small), idx, f))
        scored.sort(key=lambda t: (-t[0], t[1]))
        for _, _, f in scored[: min(4, len(scored))]:
            for line in _ocr_lines(f):
                if line not in ocr_lines:
                    ocr_lines.append(line)

    audio = _audio_features(clip_path)
    scene_tags = _scene_tags(metrics, audio)
    ocr_toks = _ocr_tokens(ocr_lines, max_tokens=4)

    description = _description(obj_stats, ocr_lines, audio, metrics)
    keywords = _compose_keywords(obj_stats, scene_tags, ocr_toks)

    # Evidence for downstream trust/UX
    # Sort objects by a “stability” score (presence then counts)
    obj_items = []
    for label, total in obj_stats.total_counts.items():
        pres = obj_stats.frame_presence.get(label, 0)
        conf = obj_stats.avg_conf.get(label, 0.0)
        obj_items.append((pres, total, conf, label))
    obj_items.sort(key=lambda t: (-t[0], -t[1], -t[2], t[3]))

    evidence = {
        "objects": [
            {"label": label, "frames_present": pres, "total_detections": total, "avg_conf": round(float(conf), 3)}
            for pres, total, conf, label in obj_items
        ],
        "ocr_lines": ocr_lines[:8],
        "ocr_tokens": ocr_toks,
        "audio": audio,
        "scene_tags": scene_tags,
        "metrics": metrics,
    }

    return AnalyzeClipResponse(description=description, keywords=keywords, evidence=evidence)