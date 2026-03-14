"""
Feature extraction for both input modes:
  1. CSV  — exact GazeBase pipeline, full 16 clean features
  2. Image — approximate features from a webcam frame using OpenCV Haar cascades

The webcam path fills temporal features (velocity, blink rate) with population
medians from the training data checkpoint, since those require time-series signal.
"""

import numpy as np
import pandas as pd
import cv2

SAMPLE_RATE = 1000   # GazeBase recording frequency (Hz)
MAX_VEL     = 700    # physiological velocity cap (deg/s)
LAB_FIX     = 1      # GazeBase fixation label
LAB_SAC     = 2      # GazeBase saccade label

# Features the model expects (order must match scaler training order)
FEATURE_COLS = [
    "blink_rate_pm", "blink_count", "mean_blink_ms",
    "num_fixations",  "mean_fix_ms",  "std_fix_ms",
    "num_saccades",   "mean_sac_ms",  "mean_sac_vel", "peak_sac_vel", "vel_std",
    "mean_pupil",     "std_pupil",    "pupil_range",
    "gaze_x_std",     "gaze_y_std",
]


# ── Internal helpers ───────────────────────────────────────────────────────

def _run_lengths(mask) -> list:
    """Convert boolean mask into list of consecutive-True run lengths."""
    lengths, count = [], 0
    for val in mask:
        if val:
            count += 1
        else:
            if count:
                lengths.append(count)
            count = 0
    if count:
        lengths.append(count)
    return lengths


def _velocity(x, y) -> np.ndarray:
    """Frame-to-frame velocity in deg/s, capped at MAX_VEL."""
    dx = np.diff(x, prepend=x[0])
    dy = np.diff(y, prepend=y[0])
    return np.clip(np.sqrt(dx**2 + dy**2) * SAMPLE_RATE, 0, MAX_VEL)


def _safe(val) -> float:
    try:
        v = float(val)
        return round(v, 4) if not (np.isnan(v) or np.isinf(v)) else 0.0
    except Exception:
        return 0.0


# ── CSV pipeline (exact, no approximation) ────────────────────────────────

def extract_features_from_csv(df: pd.DataFrame) -> dict | None:
    """
    Extract all 16 oculomotor features from a GazeBase-format CSV dataframe.
    Returns None if fewer than 50 valid samples exist.

    Expected columns: n, x, y, val, dP, lab
    """
    valid = df[df["val"] == 0]
    if len(valid) < 50:
        return None

    x    = valid["x"].values.astype(float)
    y    = valid["y"].values.astype(float)
    dp   = valid["dP"].values.astype(float)
    dur  = len(valid) / SAMPLE_RATE
    vel  = _velocity(x, y)
    dp_pos = dp[dp > 0]

    # ── Blink: contiguous invalid-sample bursts of 50–500 ms ──────────────
    invalid = (df["val"] != 0).values
    blink_count, blink_dur = 0, []
    in_blink, blink_start = False, 0
    for i, bad in enumerate(invalid):
        if bad and not in_blink:
            in_blink, blink_start = True, i
        elif not bad and in_blink:
            d = i - blink_start
            if 50 <= d <= 500:
                blink_count += 1
                blink_dur.append(d)
            in_blink = False

    # ── Fixation / saccade: ground-truth labels when available ────────────
    has_lab  = "lab" in valid.columns and valid["lab"].notna().any()
    lab_vals = set(valid["lab"].dropna().unique()) if has_lab else set()
    use_lab  = has_lab and (LAB_FIX in lab_vals) and (LAB_SAC in lab_vals)

    if use_lab:
        fix_segs = _run_lengths((valid["lab"] == LAB_FIX).values)
        sac_segs = _run_lengths((valid["lab"] == LAB_SAC).values)
    else:
        fix_segs = _run_lengths(vel < 30)
        sac_segs = _run_lengths((vel >= 30) & (vel < MAX_VEL))

    high_vel = vel[(vel >= 30) & (vel < MAX_VEL)]

    return {
        "blink_rate_pm": _safe(blink_count / dur * 60) if dur else 0.0,
        "blink_count":   blink_count,
        "mean_blink_ms": _safe(np.mean(blink_dur))   if blink_dur     else 0.0,
        "num_fixations": len(fix_segs),
        "mean_fix_ms":   _safe(np.mean(fix_segs))    if fix_segs      else 0.0,
        "std_fix_ms":    _safe(np.std(fix_segs))     if fix_segs      else 0.0,
        "num_saccades":  len(sac_segs),
        "mean_sac_ms":   _safe(np.mean(sac_segs))    if sac_segs      else 0.0,
        "mean_sac_vel":  _safe(np.mean(high_vel))    if len(high_vel) else 0.0,
        "peak_sac_vel":  _safe(np.max(high_vel))     if len(high_vel) else 0.0,
        "vel_std":       _safe(np.std(vel)),
        "mean_pupil":    _safe(np.mean(dp_pos))      if len(dp_pos)   else 0.0,
        "std_pupil":     _safe(np.std(dp_pos))       if len(dp_pos)   else 0.0,
        "pupil_range":   _safe(np.ptp(dp_pos))       if len(dp_pos)   else 0.0,
        "gaze_x_std":    _safe(np.std(x)),
        "gaze_y_std":    _safe(np.std(y)),
    }


# ── Webcam image pipeline (approximate) ───────────────────────────────────

def extract_features_from_image(image_rgb: np.ndarray, pop_stats: dict) -> tuple[dict, str]:
    """
    Estimate oculomotor features from a single webcam frame.

    Temporal features (velocity, blink rate, etc.) cannot be measured from
    a static image and are filled with population medians.  Only pupil-size
    proxy and eye-openness ratio are derived from the image itself.

    Returns:
        (features dict, status message describing what was detected)
    """
    # Start with population medians as the safe baseline for every feature
    features = {col: float(pop_stats.get(col, {}).get("median", 0.0))
                for col in FEATURE_COLS}

    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY) \
           if len(image_rgb.shape) == 3 else image_rgb.copy()

    # ── Face detection (used to scale search region for eyes) ─────────────
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    eye_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_eye.xml"
    )

    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
    )

    if len(faces) == 0:
        # Fall back to searching the whole image for eyes
        search_region = gray
        offset_x, offset_y = 0, 0
        status = "No face detected — using full-image eye search. " \
                 "Pupil and eye metrics estimated from population median."
    else:
        fx, fy, fw, fh = sorted(faces, key=lambda f: f[2] * f[3])[-1]  # largest face
        search_region = gray[fy:fy + fh, fx:fx + fw]
        offset_x, offset_y = fx, fy
        status = "Face detected."

    eyes = eye_cascade.detectMultiScale(
        search_region, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20)
    )

    if len(eyes) == 0:
        status += " No eyes detected — all features derived from population median."
        return features, status

    # ── Analyse detected eyes ─────────────────────────────────────────────
    eye_metrics = []
    for ex, ey, ew, eh in eyes[:2]:    # at most 2 eyes
        eye_roi = gray[
            offset_y + ey : offset_y + ey + eh,
            offset_x + ex : offset_x + ex + ew,
        ]
        if eye_roi.size == 0:
            continue

        # Openness ratio: height / width (normal ~0.35–0.50, fatigued = lower)
        openness = eh / max(ew, 1)

        # Pupil proxy: darker region = larger pupil
        # Invert brightness to get pupil-size proxy in 0–1 range
        mean_brightness = float(np.mean(eye_roi))
        pupil_proxy = (255.0 - mean_brightness) / 255.0

        eye_metrics.append({"openness": openness, "pupil_proxy": pupil_proxy})

    if not eye_metrics:
        return features, status + " Eye regions were empty."

    avg_openness    = float(np.mean([m["openness"]    for m in eye_metrics]))
    avg_pupil_proxy = float(np.mean([m["pupil_proxy"] for m in eye_metrics]))

    # ── Map image metrics to feature-space proxies ────────────────────────
    # Pupil: scale proxy to GazeBase pupil diameter range (~100–800 units)
    med_pupil = float(pop_stats.get("mean_pupil", {}).get("median", 400))
    features["mean_pupil"] = med_pupil * (0.5 + avg_pupil_proxy)

    # Blink rate proxy from eye openness
    # Very open (> 0.45) → alert, less blinking
    # Very narrow (< 0.30) → fatigued, more / longer blinks
    if avg_openness > 0.45:
        features["blink_rate_pm"] = float(
            pop_stats.get("blink_rate_pm", {}).get("p25", 10)
        )
    elif avg_openness < 0.30:
        features["blink_rate_pm"] = float(
            pop_stats.get("blink_rate_pm", {}).get("p75", 20)
        )
    # Between 0.30–0.45: leave at median (already set)

    # Gaze dispersion proxy from eye position spread (if 2 eyes detected)
    if len(eyes) >= 2:
        centers = [(ex + ew / 2, ey + eh / 2) for ex, ey, ew, eh in eyes[:2]]
        features["gaze_x_std"] = abs(centers[0][0] - centers[1][0]) * 0.1
        features["gaze_y_std"] = abs(centers[0][1] - centers[1][1]) * 0.1

    n_eyes = len([m for m in eye_metrics])
    status += (
        f" {n_eyes} eye(s) detected. "
        f"Openness={avg_openness:.2f}, pupil proxy={avg_pupil_proxy:.2f}. "
        f"Temporal features (velocity, saccades) from population median."
    )

    return features, status
