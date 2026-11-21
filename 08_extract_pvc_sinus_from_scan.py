"""
08_extract_pvc_sinus_from_scan.py
---------------------------------
Extract, from ONE scanned 12-lead EP-recorder printout (3 rows x 4 columns +
rhythm strip), for every lead panel:

    * the PVC beat window
    * the immediately preceding SINUS (normal) beat window
    * scale-preserving crops (no anisotropic resize)  -> PNG
    * simple morphology numbers (duration, R/S extents)  -> CSV

Layout assumed (matches the sample scan 2_3.jpeg):
    columns: [I, II, III] [aVR, aVL, aVF] [V1, V2, V3] [V4, V5, V6]  (read as grid)
    grid rows:  I   aVR  V1  V4
                II  aVL  V2  V5
                III aVF  V3  V6
    + rhythm strip (lead II) below.   Each panel = one 2.5 s window that
    contains a PVC; the 3 leads in a column are SIMULTANEOUS.

IMPORTANT ASSUMPTIONS (please confirm / edit in CONFIG):
    * PANEL_MS = 2500  (panel width = 2.5 s, from the manuscript)
    * Voltage gain is NOT known from the scan (no calibration pulse): amplitudes
      are kept in pixels; use ratios (R/S) or per-scan normalisation.
    * Coloured vertical lines (green = sinus beat, red = PVC) drawn by the
      cardiologist are USED when present (they define the windows) and are
      REMOVED from the image before extracting the trace.
    * If no annotation lines exist for a column, an automatic detector is
      used (see detect_beats).  Always inspect the QC image.

Usage:
    python 08_extract_pvc_sinus_from_scan.py scan.jpeg --patient P2 --out OUT_DIR
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage as ndi
from scipy.signal import find_peaks
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist, squareform
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------
PANEL_MS = 2500.0
REF_W = 2576             # all pixel constants below were tuned on 2576-px-wide scans.
                         # Any other scan width is rescaled to this width first (see process_scan).
SHARED_TIME_AXIS = True  # the EP printout shows ONE 2.5 s window for all 12 leads (every column has
                         # the same beat timing). Cardiologist lines drawn in ANY panel therefore
                         # apply to ALL 12 leads, and automatic detection uses all leads jointly.
GRID_LEADS = [["I", "aVR", "V1", "V4"],
              ["II", "aVL", "V2", "V5"],
              ["III", "aVF", "V3", "V6"]]
INK_THR_STEPS = (215, 222, 229, 236, 242)   # tried in order until the trace has < 8% gaps
TICK_MARGIN = 40         # px at top/bottom of a panel ignored (frame + time ticks); beats that
                         # reach this zone are flagged 'clipped' instead of being trusted
TICK_MAX_LEN = 34        # px: isolated thin vertical stubs touching the frame are time ticks
COLOR_SAT = 40           # saturation above which a pixel is an annotation colour
EDGE_PX = 10             # columns ignored at both panel borders (frame remnants)
MATCH_MS = 200.0         # windows closer than this (start-to-start) are considered the same beat in QC
CROP_PRE_MS = 40.0       # crops start this long BEFORE the annotated window start, so that the QRS of leads
                         # whose onset precedes lead I's onset (V4-V6 often) is not cut off
CANVAS_W = 192           # source px of the scale-preserving crop (width, ~880 ms)
CANVAS_H = 384           # source px (height) = +-2.2 mV at 1 mV/cm, centred on the baseline
OUT_W, OUT_H = 64, 128   # final image size (isotropic: exactly CANVAS/3)
SINUS_MS = 360.0         # window length for automatic sinus windows (cardiologist mean ~360 ms)
PVC_MS = 440.0           # window length for automatic PVC windows  (cardiologist: 439 ms)
# onset detection, calibrated on ONE annotated scan (2_3.jpeg): sinus onsets agree
# within +-10 ms, PVC onsets are marked ~40-60 ms earlier by the cardiologist.
# RE-CALIBRATE on more annotated scans before trusting automatic windows.
ONSET_FRAC_SINUS = 0.20
ONSET_FRAC_PVC = 0.10
PVC_ONSET_SHIFT_MS = -35.0
SINUS_ONSET_SHIFT_MS = 0.0
# Which of the two shape clusters is the PVC? Width alone can swap the roles in some scans.
# With the timing prior the PVC cluster must also be the one that arrives EARLY (shorter
# preceding interval). It has no free parameter. Set False to reproduce the width-only rule.
USE_TIMING_PRIOR = True


# ------------------------------------------------------------------
# 1) Panel geometry: deskew + fit a REGULAR grid (printer layout is constant)
# ------------------------------------------------------------------
PITCH_X, PITCH_Y, PANEL_H = 543.6, 402.0, 346.0


def deskew(bgr):
    """Rotate the scan so the frame lines are horizontal/vertical."""
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    m = (g < 225).astype(np.uint8) * 255
    m[1300:, :] = 0
    hang, vang = [], []
    L = cv2.HoughLinesP(m, 1, np.pi / 1800, threshold=300, minLineLength=450, maxLineGap=15)
    if L is not None:
        hang = [np.degrees(np.arctan2(y2 - y1, x2 - x1)) for x1, y1, x2, y2 in L.reshape(-1, 4)]
        hang = [a for a in hang if abs(a) < 8]
    L = cv2.HoughLinesP(m, 1, np.pi / 1800, threshold=250, minLineLength=250, maxLineGap=15)
    if L is not None:
        vang = [np.degrees(np.arctan2(x2 - x1, y2 - y1)) for x1, y1, x2, y2 in L.reshape(-1, 4)]
        vang = [a for a in vang if abs(a) < 8]
    h = np.median(hang) if len(hang) >= 10 else None
    v = np.median(vang) if len(vang) >= 10 else None
    if h is not None and v is not None:
        theta = (h - v) / 2
    elif h is not None:
        theta = h
    elif v is not None:
        theta = -v
    else:
        theta = 0.0
    if abs(theta) < 0.15:
        return bgr, 0.0
    H, W = g.shape
    M = cv2.getRotationMatrix2D((W / 2, H / 2), theta, 1.0)
    return cv2.warpAffine(bgr, M, (W, H), flags=cv2.INTER_LINEAR,
                          borderValue=(255, 255, 255)), float(theta)


def fit_grid(bgr):
    """Return (x0, y0): top-left of the first panel frame. The 3x4 grid has
    constant pitch, so only the offset has to be estimated."""
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    sat = bgr.max(2).astype(int) - bgr.min(2).astype(int)
    m = ((g < 225) & (sat < 35)).astype(np.uint8) * 255
    m[1330:, :] = 0
    V = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 130)))
    Hh = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (300, 1)))
    vx = (V > 0).sum(0).astype(float)
    hy = (Hh > 0).sum(1).astype(float)
    W = vx.size
    best = (-1, 0)
    for x0 in range(150, 400):
        sc = sum(vx[int(round(x0 + j * PITCH_X)) - 2:int(round(x0 + j * PITCH_X)) + 3].max()
                 for j in range(5) if int(round(x0 + j * PITCH_X)) + 3 < W)
        if sc > best[0]:
            best = (sc, x0)
    x0 = best[1]
    best = (-1, 0)
    for y0 in range(40, 300):
        sc = 0
        for k in range(3):
            for yy in (y0 + k * PITCH_Y, y0 + k * PITCH_Y + PANEL_H):
                yy = int(round(yy))
                sc += hy[yy - 2:yy + 3].max()
        if sc > best[0]:
            best = (sc, y0)
    return x0, best[1]


# ------------------------------------------------------------------
# 2) Trace extraction
# ------------------------------------------------------------------
def _ink_mask(rgb, gray, x0, x1, y0, y1, thr):
    sat = rgb.max(2) - rgb.min(2)
    sl = (slice(y0 + TICK_MARGIN, y1 - TICK_MARGIN), slice(x0 + 4, x1 - 4))
    ink = (gray[sl] < thr) & ~(sat[sl] > COLOR_SAT)
    lab, n = ndi.label(ink, structure=np.ones((3, 3)))
    keep = np.zeros_like(ink)
    Hh = ink.shape[0]
    for i, o in enumerate(ndi.find_objects(lab), start=1):
        h = o[0].stop - o[0].start
        w = o[1].stop - o[1].start
        if max(h, w) < 10:                        # specks
            continue
        touches_edge = o[0].start <= 2 or o[0].stop >= Hh - 2
        if touches_edge and w <= 5 and h <= TICK_MAX_LEN:
            continue                              # time tick attached to the frame line
        keep[o] |= (lab[o] == i)
    return keep


def extract_panel_trace(rgb, gray, x0, x1, y0, y1, line_xs=()):
    """Return dict with per-column ymin/ymax/center in panel coordinates.
    The ink threshold is adapted per panel: faint (light-printed) traces need a
    higher threshold. 'gap_frac' = fraction of columns with no ink before gap
    interpolation (a panel with a large gap_frac is unreliable)."""
    for thr in INK_THR_STEPS:
        keep = _ink_mask(rgb, gray, x0, x1, y0, y1, thr)
        cols = keep.any(0)
        cols[:EDGE_PX] = True
        cols[-EDGE_PX:] = True
        gap = float(1 - cols.mean())
        if gap < 0.08:
            break
    ncol = keep.shape[1]
    ymin = np.full(ncol, np.nan)
    ymax = np.full(ncol, np.nan)
    rows = np.arange(keep.shape[0])
    for c in range(ncol):
        r = rows[keep[:, c]]
        if len(r):
            ymin[c], ymax[c] = r.min(), r.max()
    edge_touch = (np.nan_to_num(ymin, nan=99) <= 1) | (np.nan_to_num(ymax, nan=-1) >= keep.shape[0] - 2)
    # frame / edge remnants: do not trust the first and last EDGE_PX columns
    for arr in (ymin, ymax):
        arr[:EDGE_PX] = np.nan
        arr[-EDGE_PX:] = np.nan
    # annotation lines that cross the trace leave residual dark pixels: in those
    # columns keep only ink inside the envelope of the neighbouring columns
    for lx in line_xs:
        lx = int(round(lx - 4))                     # panel x -> sub-image x
        cols = [c for c in range(lx - 3, lx + 4) if 0 <= c < ncol]
        nb = [c for c in list(range(lx - 9, lx - 4)) + list(range(lx + 5, lx + 10))
              if 0 <= c < ncol and not np.isnan(ymin[c])]
        if not nb:
            continue
        lo = min(ymin[c] for c in nb) - 4
        hi = max(ymax[c] for c in nb) + 4
        for c in cols:
            r = rows[keep[:, c] & (rows >= lo) & (rows <= hi)]
            if len(r):
                ymin[c], ymax[c] = r.min(), r.max()
            else:
                ymin[c] = ymax[c] = np.nan
    # interpolate small gaps (e.g. where an annotation line crossed the trace)
    idx = np.arange(ncol)
    for arr in (ymin, ymax):
        ok = ~np.isnan(arr)
        if ok.sum() > 10:
            arr[:] = np.interp(idx, idx[ok], arr[ok])
    center = (ymin + ymax) / 2.0
    return {"ymin": ymin, "ymax": ymax, "center": center, "offset_y": y0 + TICK_MARGIN,
            "offset_x": x0 + 4, "gap_frac": gap, "ink_thr": thr,
            "edge_touch": edge_touch}


# ------------------------------------------------------------------
# 3) Annotation lines (cardiologist) -> windows
# ------------------------------------------------------------------
def read_annotation_lines(rgb, x0, x1, y0, y1):
    """Detect red and green vertical lines inside a panel (each colour handled
    separately so touching red/green lines are not merged). Returns a list of
    (x_in_panel, 'green'|'red')."""
    sub = rgb[y0:y1, x0:x1].astype(int)
    r, g, b = sub[..., 0], sub[..., 1], sub[..., 2]
    sat = sub.max(2) - sub.min(2)
    masks = {"red": (r > g + 30) & (r > b + 30) & (sat > COLOR_SAT),
             "green": (g > r + 12) & (g > b + 8) & (sat > COLOR_SAT)}
    H = y1 - y0
    out = []
    for colour, m in masks.items():
        cnt = m.sum(0).astype(float)
        sm = np.convolve(cnt, np.ones(7), mode="same")          # tolerate tilt
        cand = np.where(sm > 0.35 * H)[0]
        if len(cand) == 0:
            continue
        for grp in np.split(cand, np.where(np.diff(cand) > 3)[0] + 1):
            # centre of colour mass in this group, weighted by pixel counts
            w = cnt[max(0, grp[0] - 3):grp[-1] + 4]
            xs = np.arange(max(0, grp[0] - 3), max(0, grp[0] - 3) + len(w))
            out.append((float((xs * w).sum() / max(w.sum(), 1)), colour))
    return sorted(out)


def windows_from_lines(lines):
    """Cardiologist lines come in consecutive (start, end) pairs sorted by x.
    A pair is a PVC window if at least one of its two lines is red, otherwise a
    sinus window (colours are not perfectly consistent between scans)."""
    xs = sorted(lines, key=lambda t: t[0])
    wins = []
    for i in range(0, len(xs) - 1, 2):
        (x1, c1), (x2, c2) = xs[i], xs[i + 1]
        if x2 - x1 < 20:               # duplicated/thick line, not a window
            continue
        kind = "PVC" if "red" in (c1, c2) else "sinus"
        wins.append({"kind": kind, "start": x1, "end": x2})
    return wins


# ------------------------------------------------------------------
# 4) Automatic detector (used when no annotation lines exist)
# ------------------------------------------------------------------
def activity(trace):
    """Vertical extent of ink per x column: large on steep QRS slopes."""
    ext = (trace["ymax"] - trace["ymin"])
    ext = np.nan_to_num(ext)
    return ext


def _beat_vector(traces_col, p, px_per_ms, n_out=60):
    """Shape of the beat around peak p over all simultaneous leads, each lead
    normalised by its own amplitude range so every lead weighs the same."""
    lo, hi = int(p - 100 * px_per_ms), int(p + 260 * px_per_ms)
    vecs = []
    for t in traces_col:
        c = np.nan_to_num(t["center"], nan=np.nanmedian(t["center"]))
        scale = np.percentile(c, 98) - np.percentile(c, 2) + 1e-6
        pad = max(0, -lo, hi - len(c))
        cc = np.pad(c, (pad, pad), mode="edge")
        seg = cc[lo + pad:hi + pad]
        seg = (seg - np.median(seg)) / scale
        vecs.append(np.interp(np.linspace(0, len(seg) - 1, n_out), np.arange(len(seg)), seg))
    return np.concatenate(vecs)


def classify_beats(traces_col, peaks, wlen, score, px_per_ms):
    """Boolean array: is the beat a PVC?
    1) split beats into two shape clusters (sinus beats resemble each other),
    2) the cluster with the wider QRS is the PVC cluster,
    3) fall back to the single widest/largest beat if no clear split exists
       (every recording is known to contain at least one PVC)."""
    n = len(peaks)
    fallback = score == score.max()
    if n < 2:
        return fallback
    V = np.stack([_beat_vector(traces_col, p, px_per_ms) for p in peaks])
    if n == 2:
        ratio = max(wlen) / max(min(wlen), 1e-6)
        return (wlen == wlen.max()) if ratio > 1.15 else fallback
    D = squareform(pdist(V))
    lab = fcluster(linkage(pdist(V), "average"), 2, "maxclust") - 1
    if len(set(lab)) < 2:
        return fallback
    within = np.mean([D[i, j] for i in range(n) for j in range(i + 1, n) if lab[i] == lab[j]]
                     or [1e-6])
    between = np.mean([D[i, j] for i in range(n) for j in range(n) if lab[i] != lab[j]])
    if between / max(within, 1e-6) < 1.3:
        return fallback
    w0, w1 = wlen[lab == 0].mean(), wlen[lab == 1].mean()
    pvc_cluster = 0 if w0 > w1 else 1
    if USE_TIMING_PRIOR:
        t = np.asarray(peaks, float) / px_per_ms
        pre = np.r_[np.nan, np.diff(t)]                    # interval before each beat

        def mean_pre(c):
            x = pre[(lab == c) & ~np.isnan(pre)]
            return x.mean() if len(x) else np.nan
        pa, pb = mean_pre(0), mean_pre(1)
        if not (np.isnan(pa) or np.isnan(pb)) and pa > 0 and pb > 0:
            # evidence that cluster 0 is the PVC: wider QRS  +  shorter preceding interval
            s0 = np.log(w0 / max(w1, 1e-6)) + np.log(pb / pa)
            pvc_cluster = 0 if s0 > 0 else 1
    return lab == pvc_cluster


def detect_beats(traces_col, px_per_ms):
    """traces_col: list of the simultaneous leads of one column. Returns list of
    dict(kind,start,end) in panel x coordinates. A column can hold several PVCs:
    beats are split into sinus / PVC groups by shape similarity + QRS width."""
    acts = []
    for t in traces_col:
        a = activity(t)
        sc = np.percentile(a, 99) + 1e-6
        acts.append(np.clip(a / sc, 0, 1))
    A = np.sum(acts, 0)
    k = max(3, int(30 * px_per_ms))
    A = np.convolve(A, np.ones(k) / k, mode="same")
    base = float(np.median(A))
    # >= 330 ms apart: merges the R and S deflections of one wide PVC into one beat
    peaks, _ = find_peaks(A, height=base + 0.3 * (A.max() - base),
                          distance=int(330 * px_per_ms))
    # a beat cut by the panel border is neither measurable nor croppable: drop it
    peaks = np.array([p for p in peaks
                      if int(60 * px_per_ms) <= p <= len(A) - int(150 * px_per_ms)], dtype=int)
    if len(peaks) == 0:
        return []
    reach = int(250 * px_per_ms)

    def span(p, frac):
        """left/right edge where activity falls below base + frac*(peak-base),
        searched within +-250 ms (prevents running away on noisy baselines)."""
        thr = base + frac * (A[p] - base)
        l = p
        while l > max(0, p - reach) and A[l] > thr:
            l -= 1
        r = p
        while r < min(len(A) - 1, p + reach) and A[r] > thr:
            r += 1
        return l, r

    # excursion duration: how long the signal stays away from baseline (summed over
    # leads, each lead normalised). Wide/slow PVCs stay away longer than sinus beats;
    # steepness-based activity underestimates slow deflections.
    Dv = np.zeros(len(A))
    for t in traces_col:
        c = np.nan_to_num(t["center"], nan=np.nanmedian(t["center"]))
        sc = np.percentile(c, 98) - np.percentile(c, 2) + 1e-6
        Dv += np.abs(c - np.median(c)) / sc
    Dv = np.convolve(Dv, np.ones(k) / k, mode="same")
    dbase = float(np.median(Dv))

    def excursion_ms(p, frac=0.25):
        lo, hi = max(0, p - int(120 * px_per_ms)), min(len(Dv), p + int(320 * px_per_ms))
        pk = Dv[lo:hi].max()
        thr = dbase + frac * (pk - dbase)
        above = Dv[lo:hi] > thr
        i = int(np.argmax(Dv[lo:hi]))
        l = i
        while l > 0 and above[l - 1]:
            l -= 1
        r = i
        while r < len(above) - 1 and above[r + 1]:
            r += 1
        return (r - l + 1) / px_per_ms

    wlen = np.array([excursion_ms(p) for p in peaks])
    amps = np.array([sum(np.nanmax(t["ymax"][max(0, p - int(80 * px_per_ms)):p + int(80 * px_per_ms)])
                         - np.nanmin(t["ymin"][max(0, p - int(80 * px_per_ms)):p + int(80 * px_per_ms)])
                         for t in traces_col) for p in peaks])
    score = wlen / (wlen.max() + 1e-9) + 0.5 * amps / (amps.max() + 1e-9)
    is_pvc = classify_beats(traces_col, peaks, wlen, score, px_per_ms)
    beats = []
    for p, pv in zip(peaks, is_pvc):
        if pv:
            onset = max(0, span(p, ONSET_FRAC_PVC)[0] + int(PVC_ONSET_SHIFT_MS * px_per_ms))
            length, kind = PVC_MS * px_per_ms, "PVC"
        else:
            onset = max(0, span(p, ONSET_FRAC_SINUS)[0] + int(SINUS_ONSET_SHIFT_MS * px_per_ms))
            length, kind = SINUS_MS * px_per_ms, "sinus"
        beats.append({"kind": kind, "start": float(onset), "end": float(onset + length),
                      "peak": int(p), "width_ms": float(wlen[list(peaks).index(p)])})
    return beats


# ------------------------------------------------------------------
# 5) Crops and measurements
# ------------------------------------------------------------------
def render_crop(trace, x_start, x_end, panel_w, y_scale=1.0):
    """Scale-preserving canvas: white image CANVAS_W x CANVAS_H, baseline at the
    vertical centre, window starting at x=20 px. y_scale = gain (mV/cm) of the scan,
    so that the vertical axis of EVERY crop is the same physical voltage scale
    (reference: 1.0 mV/cm). Returns uint8 image + clipped flag."""
    xs = np.arange(int(round(x_start)), int(round(x_end)))
    xs = xs[(xs >= 0) & (xs < len(trace["center"]))]
    canvas = np.full((CANVAS_H, CANVAS_W), 255, dtype=np.uint8)
    if len(xs) < 5:
        return canvas, True, np.nan
    # baseline = median of centre line over the window
    base = np.nanmedian(trace["center"][xs])
    clipped = False
    for i, x in enumerate(xs):
        cx = 20 + i
        if cx >= CANVAS_W:
            break
        y_top = (trace["ymin"][x] - base) * y_scale + CANVAS_H / 2
        y_bot = (trace["ymax"][x] - base) * y_scale + CANVAS_H / 2
        if y_top < 0 or y_bot >= CANVAS_H:
            clipped = True
        t, b = int(np.clip(round(y_top), 0, CANVAS_H - 1)), int(np.clip(round(y_bot), 0, CANVAS_H - 1))
        canvas[t:b + 1, cx] = 0
    # thicken slightly so downsampling keeps the line
    canvas = ndi.minimum_filter(canvas, size=2)
    return canvas, clipped, float(base)


def measure(trace, x_start, x_end, px_per_ms, mv_per_px):
    xs = np.arange(int(round(x_start)), int(round(x_end)))
    xs = xs[(xs >= 0) & (xs < len(trace["center"]))]
    if len(xs) < 5:
        return {}
    base = np.nanmedian(trace["center"][xs])
    up = base - np.nanmin(trace["ymin"][xs])        # px above baseline
    down = np.nanmax(trace["ymax"][xs]) - base      # px below baseline
    act = activity({"ymin": trace["ymin"][xs], "ymax": trace["ymax"][xs]})
    qrs_cols = int((act > 0.25 * act.max()).sum()) if act.max() > 0 else 0
    return {"win_ms": len(xs) / px_per_ms,
            "pos_mV": float(up * mv_per_px), "neg_mV": float(down * mv_per_px),
            "pos_neg_ratio": float(up / (down + 1e-6)), "active_ms": qrs_cols / px_per_ms}


def qrs_offset_ms(trace, start, end, px_per_ms, look_back_ms=100.0):
    """Time of the steepest activity (QRS) relative to the window start, in ms.
    Searched from look_back_ms BEFORE the start, so a negative value means the QRS of this
    lead lies outside (before) the window."""
    lo = max(0, int(round(start - look_back_ms * px_per_ms)))
    hi = min(len(trace["center"]), int(round(end)))
    if hi - lo < 10:
        return np.nan
    a = activity({"ymin": trace["ymin"][lo:hi], "ymax": trace["ymax"][lo:hi]})
    k = max(3, int(10 * px_per_ms))
    a = np.convolve(a, np.ones(k) / k, mode="same")
    return float((lo + int(np.argmax(a)) - start) / px_per_ms)


def panel_is_corrupt(rgb, gray, x0, x1, y0, y1, trace):
    """Scanner/recorder noise blocks (black rectangles, dense oscillation)."""
    sub = gray[y0 + TICK_MARGIN:y1 - TICK_MARGIN, x0 + 4:x1 - 4]
    dark_frac = float((sub < 140).mean())
    ext = np.nan_to_num(trace["ymax"] - trace["ymin"])
    busy = float((ext > 25).mean())            # fraction of columns with tall ink
    empty = float(np.isnan(trace["center"]).mean())
    gappy = trace.get("gap_frac", 0.0) > 0.15
    return (dark_frac > 0.15) or (busy > 0.6) or gappy, {
        "dark_frac": dark_frac, "busy_frac": busy, "gap_frac": trace.get("gap_frac", 0.0)}


# ------------------------------------------------------------------
# 6) Main
# ------------------------------------------------------------------
def process_scan(path, patient, out_dir, label=None, gain_mv_per_cm=1.0, recording=None,
                 scan_id=None):
    out_dir = Path(out_dir)
    (out_dir / "crops").mkdir(parents=True, exist_ok=True)
    sid = scan_id or patient
    # np.fromfile + imdecode: cv2.imread fails on non-ASCII (e.g. Persian) Windows paths
    bgr = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"cannot read image: {path}")
    H0, W0 = bgr.shape[:2]
    rescale = REF_W / W0
    if abs(rescale - 1.0) > 0.005:
        bgr = cv2.resize(bgr, (REF_W, int(round(H0 * rescale))),
                         interpolation=cv2.INTER_AREA if rescale < 1 else cv2.INTER_CUBIC)
        print(f"{Path(path).name}: {W0}x{H0} rescaled x{rescale:.3f} to width {REF_W}")
    if abs(bgr.shape[0] / bgr.shape[1] - 1873 / 2576) > 0.03:
        print(f"  WARNING {Path(path).name}: aspect ratio differs from 2576x1873 -> layout may be off, check qc_windows.png")
    bgr, theta = deskew(bgr)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    gray = rgb.astype(float).mean(2)
    gx0, gy0 = fit_grid(bgr)
    xe = [int(round(gx0 + j * PITCH_X)) for j in range(5)]
    row_bounds = [(int(round(gy0 + r * PITCH_Y)), int(round(gy0 + r * PITCH_Y + PANEL_H)))
                  for r in range(3)]
    panel_w = PITCH_X
    px_per_ms = panel_w / PANEL_MS
    px_per_mm = panel_w / (PANEL_MS / 40.0)               # 25 mm/s -> 1 mm = 40 ms
    mv_per_px = gain_mv_per_cm / (10.0 * px_per_mm)
    print(f"{Path(path).name}: deskew {theta:+.2f} deg, grid origin ({gx0},{gy0}), "
          f"{1000 * px_per_ms:.0f} px/s, {px_per_mm:.2f} px/mm, gain {gain_mv_per_cm} mV/cm")

    traces, ann, corrupt, cinfo = {}, {}, {}, {}
    for r, (y0, y1) in enumerate(row_bounds):
        for c in range(4):
            x0, x1 = xe[c], xe[c + 1]
            lead = GRID_LEADS[r][c]
            raw_lines = read_annotation_lines(rgb, x0, x1, y0, y1)
            traces[lead] = extract_panel_trace(rgb, gray, x0, x1, y0, y1,
                                               line_xs=[x for x, _ in raw_lines])
            ann[lead] = windows_from_lines(raw_lines)
            corrupt[lead], cinfo[lead] = panel_is_corrupt(rgb, gray, x0, x1, y0, y1, traces[lead])

    rows, qc, auto_val = [], {}, {}
    wf = {}                                   # waveforms (mV, relative to the beat baseline) for modelling
    all_leads = [GRID_LEADS[r][c] for c in range(4) for r in range(3)]
    all_good = [l for l in all_leads if not corrupt[l]]
    shared_lines = next((ann[l] for l in all_leads if ann[l] and any(w["kind"] == "PVC" for w in ann[l])), None)
    def _trimmed(ls):
        n = min(len(traces[l]["center"]) for l in ls)
        return [{k: (v[:n] if isinstance(v, np.ndarray) else v) for k, v in traces[l].items()} for l in ls]
    shared_auto = detect_beats(_trimmed(all_good), px_per_ms) if (SHARED_TIME_AXIS and len(all_good) >= 3) else None
    for c in range(4):
        col_leads = [GRID_LEADS[r][c] for r in range(3)]
        good = [l for l in col_leads if not corrupt[l]]
        lines = next((ann[l] for l in col_leads if ann[l]), None)
        if SHARED_TIME_AXIS and shared_lines is not None:
            lines = shared_lines
        if lines is not None and any(w["kind"] == "PVC" for w in lines):
            beats, source = lines, "cardiologist_lines"
        elif SHARED_TIME_AXIS and shared_auto:
            beats, source = shared_auto, "auto_shared"
        elif good:
            beats, source = detect_beats([traces[l] for l in good], px_per_ms), "auto"
        else:
            beats, source = [], "none"
        qc[c] = (beats, source)
        if source == "cardiologist_lines" and good:           # validation only, not used for crops
            auto_val[c] = shared_auto if (SHARED_TIME_AXIS and shared_auto is not None) \
                else detect_beats([traces[l] for l in good], px_per_ms)
        pvcs = [b for b in beats if b["kind"] == "PVC"]
        sinus_all = [b for b in beats if b["kind"] == "sinus"]
        for pi, pvc in enumerate(pvcs):
            if pvc["end"] > panel_w or pvc["start"] < 0:       # truncated by panel edge
                continue
            prior = [b for b in sinus_all if b["end"] <= pvc["start"] + 5]
            follow = [b for b in sinus_all if b["start"] >= pvc["end"] - 5]
            if prior:
                sinus, spos = max(prior, key=lambda b: b["start"]), "preceding"
            elif follow:
                sinus, spos = min(follow, key=lambda b: b["start"]), "following"
            else:
                sinus, spos = None, "none"
            for ld in (all_leads if SHARED_TIME_AXIS else col_leads):
                if corrupt[ld] or (SHARED_TIME_AXIS and c > 0):
                    continue          # shared mode: crop every lead ONCE (while handling column 1)
                for kind, b in (("PVC", pvc), ("sinus", sinus)):
                    if b is None or b["end"] > panel_w:
                        continue
                    img, clipped, base = render_crop(traces[ld], max(0.0, b["start"] - CROP_PRE_MS * px_per_ms),
                                                     b["end"], panel_w, y_scale=gain_mv_per_cm)
                    w0, w1 = int(max(0, b["start"])), int(min(len(traces[ld]["edge_touch"]), b["end"]))
                    if traces[ld]["edge_touch"][w0:w1].any():
                        clipped = True           # beat reaches the printable edge of the panel
                    small = np.asarray(Image.fromarray(img).resize((OUT_W, OUT_H), Image.LANCZOS))
                    fn = f"{sid}_c{c + 1}_pvc{pi + 1}_{ld}_{kind}.png"
                    Image.fromarray(small).save(out_dir / "crops" / fn)
                    m = measure(traces[ld], b["start"], b["end"], px_per_ms, mv_per_px)
                    rows.append({"scan_id": sid, "patient_id": patient, "recording": recording, "label": label,
                                 "column": c + 1, "pvc_index": pi + 1, "lead": ld, "beat": kind,
                                 "sinus_position": spos if kind == "sinus" else "",
                                 "file": fn, "window_source": source,
                                 "start_ms": b["start"] / px_per_ms, "end_ms": b["end"] / px_per_ms,
                                 "gain_mV_per_cm": gain_mv_per_cm, "clipped": clipped,
                                 "qrs_peak_offset_ms": qrs_offset_ms(traces[ld], b["start"], b["end"], px_per_ms),
                                 **m})
                    lo_ = int(round(max(0.0, b["start"] - CROP_PRE_MS * px_per_ms)))
                    hi_ = int(round(min(len(traces[ld]["center"]), b["end"])))
                    if hi_ - lo_ >= 5:
                        w0_, w1_ = int(max(0, round(b["start"]))), hi_
                        base_ = np.nanmedian(traces[ld]["center"][w0_:w1_])
                        wf[f"{ld}|{kind}|c{c + 1}|p{pi + 1}"] = np.stack([
                            (base_ - traces[ld]["ymin"][lo_:hi_]) * mv_per_px,      # upper envelope, mV (+ = up)
                            (base_ - traces[ld]["ymax"][lo_:hi_]) * mv_per_px,      # lower envelope, mV
                        ]).astype("float32")
    df = pd.DataFrame(rows)
    # The 12 leads are simultaneous: every PVC window that starts within 150 ms of another
    # one is the SAME PVC seen in other leads -> same event_id (downstream code must treat
    # the 12 leads of one PVC jointly and must not count it several times).
    if len(df):
        pv = (df[df.beat == "PVC"].drop_duplicates(["column", "pvc_index"])
              [["column", "pvc_index", "start_ms"]].sort_values("start_ms"))
        ev_ids, cur, anchor = [], -1, -1e9
        for r_ in pv.itertuples():
            if r_.start_ms - anchor > 150:
                cur += 1
                anchor = r_.start_ms
            ev_ids.append(cur + 1)
        pv["event_id"] = ev_ids
        df = df.merge(pv[["column", "pvc_index", "event_id"]], on=["column", "pvc_index"], how="left")
    df.to_csv(out_dir / "beats_extracted.csv", index=False)
    if wf:
        np.savez_compressed(out_dir / "waveforms.npz", **wf)

    def _clean(beats):
        return [{k: (float(v) if isinstance(v, (int, float, np.integer, np.floating)) else v)
                 for k, v in b.items()} for b in beats]
    process_scan.last_info = {
        "rescale": float(rescale), "theta": float(theta), "grid_origin": [int(gx0), int(gy0)], "px_per_ms": float(px_per_ms),
        "gain": float(gain_mv_per_cm), "panel_ms": float(PANEL_MS),
        "gap_frac": {l: float(traces[l]["gap_frac"]) for l in traces},
        "unreliable": {l: bool(corrupt[l]) for l in corrupt},
        "columns": {str(c + 1): {"source": qc[c][1], "beats": _clean(qc[c][0])} for c in qc},
        "auto_for_annotated": {str(c + 1): _clean(b) for c, b in auto_val.items()},
        "n_events": int(df.event_id.nunique()) if len(df) else 0,          # UNIQUE PVC events
        "n_pvc_windows": int(df[df.beat == "PVC"].groupby(["column", "pvc_index"]).ngroups) if len(df) else 0,
        "n_crops": int(len(df)),
        "wf_dt_ms": float(1.0 / px_per_ms), "wf_pre_ms": float(CROP_PRE_MS),
        "n_clipped": int(df.clipped.sum()) if len(df) else 0,
        "leads_with_crops": sorted(df.lead.unique().tolist()) if len(df) else [],
    }

    # ---------- overlay of the windows on the real panel images (panels only, no header) ----------
    try:
        tiles = []
        for r in range(3):
            row_t = []
            for c in range(4):
                x0, x1 = xe[c], xe[c + 1]
                y0, y1 = row_bounds[r]
                tile = rgb[y0:y1, x0:x1].copy()
                ov = tile.copy()
                for b in qc.get(c, ([], ""))[0]:
                    col_ = (230, 40, 40) if b["kind"] == "PVC" else (30, 160, 60)
                    cv2.rectangle(ov, (int(b["start"]), 0), (int(b["end"]), tile.shape[0]), col_, -1)
                for b in auto_val.get(c, []):
                    cv2.rectangle(ov, (int(b["start"]), tile.shape[0] - 30), (int(b["end"]), tile.shape[0]),
                                  (120, 0, 0) if b["kind"] == "PVC" else (0, 90, 0), -1)
                tile = cv2.addWeighted(ov, 0.22, tile, 0.78, 0)
                for b in auto_val.get(c, []):          # strip drawn solid (not blended)
                    cv2.rectangle(tile, (int(b["start"]), tile.shape[0] - 30), (int(b["end"]), tile.shape[0]),
                                  (140, 0, 0) if b["kind"] == "PVC" else (0, 110, 0), -1)
                if corrupt[GRID_LEADS[r][c]]:
                    cv2.line(tile, (0, 0), (tile.shape[1], tile.shape[0]), (255, 0, 0), 6)
                cv2.putText(tile, GRID_LEADS[r][c], (8, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 160), 3)
                row_t.append(cv2.resize(tile, (520, 340)))
            tiles.append(np.hstack(row_t))
        ov_img = cv2.cvtColor(np.vstack(tiles), cv2.COLOR_RGB2BGR)
        cv2.imencode(".jpg", ov_img, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tofile(str(out_dir / "qc_overlay.jpg"))
    except Exception as e:                      # QC only: never stop the extraction
        print("  (overlay skipped:", e, ")")

    # ---------- QC figure ----------
    fig, axes = plt.subplots(3, 4, figsize=(20, 9))
    for r in range(3):
        for c in range(4):
            ld = GRID_LEADS[r][c]
            t = traces[ld]
            ax = axes[r, c]
            x = np.arange(len(t["center"])) / px_per_ms
            if corrupt[ld]:
                ax.text(0.5, 0.5, "UNRELIABLE PANEL\n(excluded)", ha="center", va="center",
                        transform=ax.transAxes, color="red", fontsize=12)
            else:
                ax.fill_between(x, -(t["ymax"]), -(t["ymin"]), color="k", lw=0)
            beats, source = qc.get(c, ([], ""))
            for b in beats:
                ax.axvspan(b["start"] / px_per_ms, b["end"] / px_per_ms, alpha=0.25,
                           color="red" if b["kind"] == "PVC" else "green")
            # when the cardiologist's lines are used as truth, ALSO show what the automatic
            # detector would have done (thin strip at the bottom of the panel): a strip whose
            # colour disagrees with the shaded window is a classification error
            n_bad = 0
            for b in auto_val.get(c, []):
                ax.axvspan(b["start"] / px_per_ms, b["end"] / px_per_ms, ymin=0.0, ymax=0.09,
                           color="darkred" if b["kind"] == "PVC" else "darkgreen", alpha=0.95, lw=0)
                for w in beats:
                    if abs(w["start"] - b["start"]) / px_per_ms < MATCH_MS and w["kind"] != b["kind"]:
                        n_bad += 1
            tag = f"  [auto strip: {n_bad} mismatch]" if (source == "cardiologist_lines" and auto_val.get(c)) else ""
            ax.set_title(f"{ld} ({source}){tag}", fontsize=9, color="red" if n_bad else "black")
    plt.suptitle(f"{patient}  label={label}  gain={gain_mv_per_cm} mV/cm  deskew={theta:+.2f} deg   "
                 f"(shaded = used windows; dark strip at the bottom = automatic detector)")
    plt.tight_layout()
    plt.savefig(out_dir / "qc_windows.png", dpi=80)
    plt.close()
    return df, qc, px_per_ms


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("scan")
    ap.add_argument("--patient", default="P0")
    ap.add_argument("--label", default=None, help="LVOT or RVOT")
    ap.add_argument("--gain", type=float, default=1.0, help="mV/cm from the printout header")
    ap.add_argument("--out", default="extracted")
    a = ap.parse_args()
    df, qc, ppm = process_scan(a.scan, a.patient, a.out, a.label, a.gain)
    print(df.groupby(["column", "window_source", "beat"]).size())
