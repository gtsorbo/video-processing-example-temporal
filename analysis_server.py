from __future__ import annotations

import os
import random
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# YOLO (ultralytics)
try:
    from ultralytics import YOLO
except Exception:
    YOLO = None


# =============================================================================
# 1) Semi-random bounded failures (all clips eventually succeed)
# =============================================================================

_failure_budget: Dict[str, int] = {}
_failure_lock = threading.Lock()


def _get_failure_budget(clip_path: Path) -> int:
    """
    Assign a random failure budget per clip once per server process.
    Budget = number of times we intentionally fail for this clip before succeeding forever.

    Env knobs:
      SIMULATE_FAILURES (default true)
      FAIL_PROB        (default 0.35) probability a clip will have failures
      FAIL_MAX         (default 2)    max number of failures for a clip that is chosen to fail
    """
    fail_prob = float(os.getenv("FAIL_PROB", "0.35"))
    fail_max = int(os.getenv("FAIL_MAX", "2"))
    key = str(clip_path)

    with _failure_lock:
        if key not in _failure_budget:
            _failure_budget[key] = random.randint(1, max(1, fail_max)) if random.random() < fail_prob else 0
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
    """
    Random-but-bounded failure simulation. All clips eventually succeed.

    Env knobs:
      SIMULATE_FAILURES (default true)
      FAIL_MODE         (default mixed) one of: mixed|500|429|slow
      SIMULATE_SLOW     (default true)
      SLOW_PROB         (default 0.15) only used for mixed
      SLOW_SLEEP_S      (default 10.0)
    """
    if os.getenv("SIMULATE_FAILURES", "true").lower() != "true":
        return

    if _get_failure_budget(clip_path) <= 0:
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


# =============================================================================
# FastAPI
# =============================================================================

app = FastAPI(title="PseudoAI.biz", version="0.6.0")


class AnalyzeClipRequest(BaseModel):
    clip_uri: str  # file://...


class AnalyzeClipResponse(BaseModel):
    keywords: List[str]  # up to 10 object labels
    evidence: Dict


def _path_from_file_uri(uri: str) -> Path:
    if not uri.startswith("file://"):
        raise HTTPException(status_code=400, detail="clip_uri must be file://...")
    return Path(uri[len("file://") :]).expanduser().resolve()


# =============================================================================
# Frame sampling (single shot input)
# =============================================================================

def _sample_frames(video_path: Path, max_frames: int) -> Tuple[List[np.ndarray], List[int]]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise HTTPException(status_code=400, detail="Could not open clip (OpenCV)")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frames: List[np.ndarray] = []
    indexes: List[int] = []

    try:
        if frame_count > 0:
            n = min(max_frames, frame_count)
            idxs = np.linspace(0, max(0, frame_count - 1), num=n, dtype=int)
            for i in idxs:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
                ok, frame = cap.read()
                if ok and frame is not None:
                    frames.append(frame)
                    indexes.append(int(i))
        else:
            i = 0
            while len(frames) < max_frames:
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                frames.append(frame)
                indexes.append(i)
                i += 1
    finally:
        cap.release()

    if not frames:
        raise HTTPException(status_code=400, detail="No frames decoded from clip")
    return frames, indexes


# =============================================================================
# YOLO detection (Step 1: richer model default; Step 2: presence-based ranking;
# Step 3: optional YOLO-World/open-vocabulary mode)
# =============================================================================

_yolo_model: Optional["YOLO"] = None
_yolo_world_classes: Optional[List[str]] = None
_yolo_world_applied: bool = False


def _parse_csv_list(env_val: str) -> List[str]:
    # comma-separated, trimmed, de-duped preserving order
    out: List[str] = []
    seen = set()
    for part in (env_val or "").split(","):
        s = part.strip()
        if not s:
            continue
        k = s.lower()
        if k in seen:
            continue
        out.append(s)
        seen.add(k)
    return out


def _load_yolo() -> "YOLO":
    """
    YOLO_MODE:
      - "coco"  (default): standard YOLOv8 COCO labels (80 classes)
      - "world": open-vocabulary style via Ultralytics YOLO-World models (requires a world-capable model + classes)

    Step 1: default to a richer model (yolov8l.pt) if YOLO_MODEL not provided.
    """
    global _yolo_model, _yolo_world_classes, _yolo_world_applied

    if _yolo_model is not None:
        return _yolo_model

    if YOLO is None:
        raise HTTPException(status_code=500, detail="ultralytics not installed (pip install ultralytics)")

    yolo_mode = os.getenv("YOLO_MODE", "coco").lower().strip()
    default_model = "yolov8l.pt" if yolo_mode != "world" else "yolov8l-world.pt"
    model_name = os.getenv("YOLO_MODEL", default_model).strip()

    _yolo_model = YOLO(model_name)

    # Step 3: Optional YOLO-World path.
    # Expect the user to provide target classes via YOLO_WORLD_CLASSES.
    if yolo_mode == "world":
        classes_env = os.getenv("YOLO_WORLD_CLASSES", "").strip()
        classes = _parse_csv_list(classes_env)
        if not classes:
            raise HTTPException(
                status_code=500,
                detail=(
                    "YOLO_MODE=world requires YOLO_WORLD_CLASSES (comma-separated list). "
                    "Example: YOLO_WORLD_CLASSES='dj,turntable,microphone,stage light,club crowd'"
                ),
            )

        # Ultralytics world-capable models typically support setting classes.
        # We do a best-effort here and fail loudly if unsupported.
        if not hasattr(_yolo_model, "set_classes"):
            raise HTTPException(
                status_code=500,
                detail=(
                    "Loaded model does not appear to support set_classes(). "
                    "Use a YOLO-World-capable model (override YOLO_MODEL) or set YOLO_MODE=coco."
                ),
            )

        try:
            _yolo_model.set_classes(classes)  # type: ignore[attr-defined]
            _yolo_world_classes = classes
            _yolo_world_applied = True
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to apply YOLO_WORLD_CLASSES via set_classes(): {e}",
            )

    return _yolo_model


def _label_map_from_result(result) -> Dict[int, str]:
    """
    result.names is usually a dict[int->str]. If it's missing, fall back to configured world classes.
    """
    global _yolo_world_classes
    names = getattr(result, "names", None)
    if isinstance(names, dict) and names:
        return {int(k): str(v) for k, v in names.items()}

    # If YOLO-World classes were set, indices should correspond to that list.
    if _yolo_world_classes:
        return {i: lbl for i, lbl in enumerate(_yolo_world_classes)}

    return {}


def _detect_object_keywords(frames: List[np.ndarray]) -> Tuple[List[str], Dict]:
    """
    Step 2: presence-based ranking
      - Count a label once per frame (if it appears in that frame)
      - Sort by frames_present desc, then name

    Returns:
      - keywords: up to 10 unique object labels
      - evidence: presence_counts + total_box_counts (for debugging)
    """
    if os.getenv("ENABLE_YOLO", "true").lower() != "true":
        return [], {"objects": {}, "note": "ENABLE_YOLO=false"}

    model = _load_yolo()
    conf_th = float(os.getenv("YOLO_CONF", "0.35"))
    max_w = int(os.getenv("DETECT_MAX_WIDTH", "960"))  # allow a bit more detail now that speed isn't a concern

    # Presence-based counts (frames where label appears)
    presence_counts = Counter()
    # Optional debug: total detections across frames
    box_counts = Counter()

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

        label_map = _label_map_from_result(r)

        frame_labels = set()
        for c in r.boxes.cls.tolist():
            cls_i = int(c)
            label = label_map.get(cls_i, str(cls_i))
            frame_labels.add(label)
            box_counts[label] += 1

        for label in frame_labels:
            presence_counts[label] += 1

    ordered = sorted(presence_counts.items(), key=lambda kv: (-kv[1], kv[0]))
    keywords = [name for name, _ in ordered[:10]]

    evidence = {
        "yolo_mode": os.getenv("YOLO_MODE", "coco").lower().strip(),
        "yolo_model": os.getenv("YOLO_MODEL", ("yolov8l.pt" if os.getenv("YOLO_MODE", "coco").lower().strip() != "world" else "yolov8l-world.pt")),
        "world_classes_applied": bool(_yolo_world_applied),
        "conf_threshold": conf_th,
        "detect_max_width": max_w,
        "frames_present_counts_top": dict(ordered[:25]),
        "box_counts_top": dict(sorted(box_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:25]),
    }
    return keywords, evidence


# =============================================================================
# Endpoint
# =============================================================================

@app.post("/analyze/clip", response_model=AnalyzeClipResponse)
def analyze_clip(req: AnalyzeClipRequest) -> AnalyzeClipResponse:
    clip_path = _path_from_file_uri(req.clip_uri)
    if not clip_path.exists() or clip_path.stat().st_size == 0:
        raise HTTPException(status_code=400, detail="Clip not found or empty")

    _maybe_fail(clip_path)

    frames, frame_indexes = _sample_frames(clip_path, max_frames=int(os.getenv("MAX_FRAMES", "10")))

    keywords, evidence = _detect_object_keywords(frames)
    evidence["frames_sampled"] = len(frames)
    evidence["frame_indexes"] = frame_indexes

    return AnalyzeClipResponse(keywords=keywords, evidence=evidence)