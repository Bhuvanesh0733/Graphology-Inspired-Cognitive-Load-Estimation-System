"""
Graphological Feature Extractor — Cognitive Load Estimation
============================================================
Extracts 15 graphological features grouped into 4 categories:

  1. PRESSURE patterns    — stroke thickness, ink density
  2. SLANT patterns       — pen angle, angular consistency
  3. SPACING patterns     — letter/word gaps, baseline drift
  4. FORM patterns        — letter shape regularity, tremor,
                            pen-lift fragmentation, zone proportions

These mirror the graphological indicators used in cognitive
load / motor-cortex research (Luria, Rosenblum et al.).
"""

import cv2
import numpy as np
from scipy.stats import entropy as scipy_entropy, skew
import warnings
warnings.filterwarnings("ignore")


# ── Preprocessing ─────────────────────────────────────────────────────────────

def preprocess(image_path):
    """
    Load image → grayscale → denoise → binary (ink = 255).
    Handles very small word-level IAM crops.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot load: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    denoised = cv2.GaussianBlur(gray, (3, 3), 0)

    # Adaptive threshold — handles uneven lighting in scanned word crops
    binary = cv2.adaptiveThreshold(
        denoised, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=15, C=8
    )
    # Sanity check: if result is mostly noise or empty, fall back to Otsu
    ink_ratio = np.sum(binary > 0) / binary.size
    if ink_ratio > 0.6 or ink_ratio < 0.01:
        _, binary = cv2.threshold(
            denoised, 0, 255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
    return gray, binary


# ── CATEGORY 1: PRESSURE ──────────────────────────────────────────────────────

def pressure_mean(binary):
    """
    Mean stroke half-width via distance transform.
    Proxy for pen pressure — thicker = more pressure.
    High cognitive load → lighter, thinner strokes.
    """
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    nz = dist[dist > 0]
    return float(np.mean(nz)) if len(nz) else 0.0


def pressure_variance(binary):
    """
    Std of stroke widths (distance transform).
    High variance = inconsistent pen pressure = cognitive load marker.
    """
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    nz = dist[dist > 0]
    return float(np.std(nz)) if len(nz) else 0.0


def ink_density(gray, binary):
    """
    Mean darkness of ink pixels (0-1 scale).
    Dense dark ink = confident strokes (low load).
    Light/faint ink = tremor or fatigue (high load).
    """
    ink_pixels = gray[binary > 0]
    if len(ink_pixels) == 0:
        return 0.0
    return float(255 - np.mean(ink_pixels)) / 255.0


# ── CATEGORY 2: SLANT ─────────────────────────────────────────────────────────

def slant_mean_angle(binary):
    """
    Mean pen slant angle across letter components.
    Graphology: consistent backward/forward slant = controlled writing.
    """
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    angles = []
    for cnt in contours:
        if cnt.shape[0] >= 5 and cv2.contourArea(cnt) > 30:
            try:
                _, _, angle = cv2.fitEllipse(cnt)
                angles.append(angle)
            except Exception:
                pass
    return float(np.mean(angles)) if angles else 90.0


def slant_deviation(binary):
    """
    Std of slant angles — core graphological load marker.
    Inconsistent slant = divided attention / high cognitive load.
    """
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    angles = []
    for cnt in contours:
        if cnt.shape[0] >= 5 and cv2.contourArea(cnt) > 30:
            try:
                _, _, angle = cv2.fitEllipse(cnt)
                angles.append(angle)
            except Exception:
                pass
    return float(np.std(angles)) if len(angles) >= 2 else 0.0


def slant_skewness(binary):
    """
    Skewness of the angle distribution.
    Asymmetric distribution → non-uniform/erratic slant.
    """
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    angles = []
    for cnt in contours:
        if cnt.shape[0] >= 5 and cv2.contourArea(cnt) > 30:
            try:
                _, _, angle = cv2.fitEllipse(cnt)
                angles.append(angle)
            except Exception:
                pass
    if len(angles) < 3:
        return 0.0
    return float(skew(angles))


# ── CATEGORY 3: SPACING ───────────────────────────────────────────────────────

def letter_spacing_mean(binary):
    """
    Mean gap between adjacent letter bounding boxes.
    Graphology: wide = relaxed; narrow = rushed/cognitively loaded.
    """
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for cnt in contours:
        if cv2.contourArea(cnt) > 15:
            x, y, w, h = cv2.boundingRect(cnt)
            boxes.append((x, x + w))
    if len(boxes) < 2:
        return 0.0
    boxes.sort(key=lambda b: b[0])
    gaps = [boxes[i+1][0] - boxes[i][1]
            for i in range(len(boxes)-1)
            if boxes[i+1][0] - boxes[i][1] > 0]
    return float(np.mean(gaps)) if gaps else 0.0


def letter_spacing_var(binary):
    """
    Variance in letter gaps.
    High variance = erratic spacing = cognitive overload.
    """
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for cnt in contours:
        if cv2.contourArea(cnt) > 15:
            x, y, w, h = cv2.boundingRect(cnt)
            boxes.append((x, x + w))
    if len(boxes) < 3:
        return 0.0
    boxes.sort(key=lambda b: b[0])
    gaps = [boxes[i+1][0] - boxes[i][1]
            for i in range(len(boxes)-1)
            if boxes[i+1][0] - boxes[i][1] > 0]
    return float(np.std(gaps)) if len(gaps) >= 2 else 0.0


def baseline_deviation(binary):
    """
    Std of y-centroids of letter components.
    Graphology: unsteady baseline = high cognitive effort.
    """
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    cy_list = []
    for cnt in contours:
        if cv2.contourArea(cnt) > 20:
            M = cv2.moments(cnt)
            if M['m00'] != 0:
                cy_list.append(M['m01'] / M['m00'])
    return float(np.std(cy_list)) if len(cy_list) >= 3 else 0.0


# ── CATEGORY 4: FORM / SHAPE ──────────────────────────────────────────────────

def form_regularity(binary):
    """
    Mean compactness (4π·area / perimeter²) of letter shapes.
    Closer to 1.0 = round/regular (low load).
    Lower = angular/irregular letters (high load).
    Graphology: 'form quality' — ideal ovals vs deformed shapes.
    """
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    compactness = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        peri = cv2.arcLength(cnt, True)
        if area > 20 and peri > 0:
            compactness.append((4 * np.pi * area) / (peri ** 2))
    return float(np.mean(compactness)) if compactness else 0.0


def tremor_index(binary):
    """
    Mean ratio of actual perimeter to convex hull perimeter.
    Ratio ≈ 1.0 = smooth strokes. Higher = jagged/tremor.
    High cognitive load → fine motor control degradation → tremor.
    """
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    ratios = []
    for cnt in contours:
        if cv2.contourArea(cnt) < 25:
            continue
        peri = cv2.arcLength(cnt, True)
        hull = cv2.convexHull(cnt)
        hull_peri = cv2.arcLength(hull, True)
        if hull_peri > 0:
            ratios.append(peri / hull_peri)
    return float(np.mean(ratios)) if ratios else 1.0


def pen_lift_fragmentation(binary):
    """
    Number of disconnected strokes per unit ink area.
    More pen lifts = hesitation/fragmentation = cognitive load.
    Graphology: 'connectedness' — linked vs broken writing.
    """
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    valid = [c for c in contours if cv2.contourArea(c) > 10]
    ink_area = max(np.sum(binary > 0), 1)
    return float(len(valid)) / (ink_area / 1000.0 + 1e-5)


def letter_height_var(binary):
    """
    Std of letter heights (bounding box h).
    Graphology: 'size regularity' — consistent height = controlled.
    High variance → letter size fluctuates = cognitive overload.
    """
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    heights = []
    for cnt in contours:
        if cv2.contourArea(cnt) > 20:
            _, _, w, h = cv2.boundingRect(cnt)
            if h > 4 and w > 2:
                heights.append(float(h))
    return float(np.std(heights)) if len(heights) >= 2 else 0.0


def pixel_entropy(gray):
    """
    Shannon entropy of the pixel intensity histogram.
    High entropy = disordered image = higher cognitive load signal.
    """
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    hist = hist / (hist.sum() + 1e-9)
    hist = hist[hist > 0]
    return float(scipy_entropy(hist))


def zone_ratio(binary):
    """
    Graphological zone analysis: upper vs lower zone ink imbalance.
    Upper zone = ascenders/loops (abstract thinking activation).
    Lower zone = descenders (motor/physical activity).
    High imbalance under cognitive load.
    """
    h = binary.shape[0]
    if h < 9:
        return 0.0
    third = h // 3
    total_ink = np.sum(binary > 0)
    if total_ink == 0:
        return 0.0
    upper_ink = np.sum(binary[:third, :] > 0)
    lower_ink = np.sum(binary[2*third:, :] > 0)
    upper_r = upper_ink / total_ink
    lower_r = lower_ink / total_ink
    return float(abs(upper_r - lower_r))


# ── Master extraction ──────────────────────────────────────────────────────────

def extract_all_features(image_path):
    """
    Extract all 15 graphological features from a single handwriting image.

    Returns:
        features_dict  : OrderedDict {name: float}
        feature_vector : numpy float32 array, length 15
    """
    gray, binary = preprocess(image_path)

    features = {
        # Pressure
        "pressure_mean":          pressure_mean(binary),
        "pressure_variance":      pressure_variance(binary),
        "ink_density":            ink_density(gray, binary),
        # Slant
        "slant_mean_angle":       slant_mean_angle(binary),
        "slant_deviation":        slant_deviation(binary),
        "slant_skewness":         slant_skewness(binary),
        # Spacing
        "letter_spacing_mean":    letter_spacing_mean(binary),
        "letter_spacing_var":     letter_spacing_var(binary),
        "baseline_deviation":     baseline_deviation(binary),
        # Form
        "form_regularity":        form_regularity(binary),
        "tremor_index":           tremor_index(binary),
        "pen_lift_fragmentation": pen_lift_fragmentation(binary),
        "letter_height_var":      letter_height_var(binary),
        "pixel_entropy":          pixel_entropy(gray),
        "zone_ratio":             zone_ratio(binary),
    }

    # Sanitise NaN / Inf
    for k, v in features.items():
        if not np.isfinite(v):
            features[k] = 0.0

    feature_vector = np.array(list(features.values()), dtype=np.float32)
    return features, feature_vector


# ── Metadata ───────────────────────────────────────────────────────────────────

FEATURE_NAMES = [
    "pressure_mean", "pressure_variance", "ink_density",
    "slant_mean_angle", "slant_deviation", "slant_skewness",
    "letter_spacing_mean", "letter_spacing_var", "baseline_deviation",
    "form_regularity", "tremor_index", "pen_lift_fragmentation",
    "letter_height_var", "pixel_entropy", "zone_ratio",
]

FEATURE_GROUPS = {
    "Pressure":   ["pressure_mean", "pressure_variance", "ink_density"],
    "Slant":      ["slant_mean_angle", "slant_deviation", "slant_skewness"],
    "Spacing":    ["letter_spacing_mean", "letter_spacing_var", "baseline_deviation"],
    "Form/Shape": ["form_regularity", "tremor_index", "pen_lift_fragmentation",
                   "letter_height_var", "pixel_entropy", "zone_ratio"],
}

# True = higher value suggests higher cognitive load (for UI colouring)
HIGH_LOAD_DIRECTION = {
    "pressure_mean":          False,
    "pressure_variance":      True,
    "ink_density":            False,
    "slant_mean_angle":       False,
    "slant_deviation":        True,
    "slant_skewness":         True,
    "letter_spacing_mean":    False,
    "letter_spacing_var":     True,
    "baseline_deviation":     True,
    "form_regularity":        False,
    "tremor_index":           True,
    "pen_lift_fragmentation": True,
    "letter_height_var":      True,
    "pixel_entropy":          True,
    "zone_ratio":             True,
}
