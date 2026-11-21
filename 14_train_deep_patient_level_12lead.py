#!/usr/bin/env python3
"""
14_train_deep_patient_level_12lead.py
-------------------------------------
12-lead version of 14_train_deep_patient_level.py (same pipeline, same three models, same patient-level CV)
with an extended, clinically oriented evaluation.

INPUT   one sample = one PVC EVENT. Channels = the scale-preserving PVC crops (images) or the extracted mV envelopes
        (waveforms) of ALL 12 leads (--leads all, default) or any subset (--leads V1,V2,V3 reproduces step 14).
        A missing lead is replaced by a blank channel.
MODELS  (0) wave1d   : tiny 1-D CNN on the 12 x (upper, lower) = 24 envelope channels
        (1) custom   : the paper's small 2-D CNN (asymmetric 3x9 first kernel), 12 input channels
        (2) mobilenet: MobileNetV2 transfer learning. ImageNet weights are 3-channel, so the first convolution is
                       adapted to 12 channels (mean of the RGB kernels, replicated and rescaled by 3/12); all other
                       layers keep the ImageNet weights. Frozen head first, then the last 20 layers (BN frozen).
EVAL    repeated stratified K-fold OVER PATIENTS (default 3 x 5); patient score = mean over his events and snapshots.
        NEW compared with step 14 (all computed on the out-of-fold patient scores):
          * confusion matrix (counts + row %), figure and CSV
          * performance at FIXED thresholds (default 0.5 = primary, plus 0.3 ... 0.7): sensitivity, specificity, PPV, NPV,
            accuracy, balanced accuracy, F1, MCC, LR+, LR-, with exact Clopper-Pearson 95 % CIs
          * calibration curve (quantile bins, Clopper-Pearson error bars), calibration slope / intercept, ECE
          * Brier score (+ patient-bootstrap CI), Brier skill score vs. the prevalence-only forecast, log-loss
          * Decision Curve Analysis (net benefit vs. treat-all / treat-none, bootstrap CI) for raw AND recalibrated scores
          * recalibrated scores = cross-validated Platt scaling of the OOF logit (2 parameters, repeated CV): the
            training uses class-balanced weights, so the RAW probabilities are centred on a 50 % prior, not on the real
            prevalence; Brier, calibration and DCA are therefore reported for raw and recalibrated scores.
XAI     Grad-CAM (now class-specific and signed: each event is explained toward its OWN class; step 14 only
        showed evidence for RVOT) and lead occlusion for every lead.

Usage
  python 14_train_deep_patient_level_12lead.py --run-dir validation_out --out deep12_out \\
        --restrict-to features_out/patient_features.csv --compare model_out2/oof_predictions.csv --ensemble 3
  python 14_train_deep_patient_level_12lead.py --run-dir validation_out --out deep12_quick --quick --no-imagenet
  python 14_train_deep_patient_level_12lead.py --selftest        # synthetic 12-lead data (null + planted signal)

Read with care: with ~54 patients the CI of an AUC is about +-0.12; step 13 already showed that adding all 12 leads
(D_all12_ratios, AUC 0.66) did worse than V1-V3, so run --leads V1,V2,V3 as the matching control and report both.
Calibration and DCA from n~54 are very unstable; treat them as descriptive, not as proof of clinical utility.
"""
import argparse
import os
import random
import shutil
import tempfile
import warnings
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from scipy import ndimage as ndi
from scipy import stats
from scipy.optimize import brentq
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (balanced_accuracy_score, confusion_matrix, log_loss, roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold

import tensorflow as tf
from tensorflow.keras import layers, Model, regularizers
from tensorflow.keras.applications import MobileNetV2

ALL_LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
CHANNEL_LEADS = list(ALL_LEADS)       # set from --leads
NCH = len(CHANNEL_LEADS)
CROP_H, CROP_W = 128, 64              # size of the stored crops
CANVAS_START_PX, PX_PER_MS = 20.0, 0.2174
PRE_MS = 40.0                         # crops start 40 ms before the annotated window
WAVE_LEN = 128                        # samples of the 1-D input (4.6 ms each, from 40 ms before the window start)
DT_MS = 1000.0 / 217.44
SNAP = 5                              # number of last epochs averaged
MB_SCALE = 2                          # MobileNetV2 gets the crops up-sampled x2


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    tf.random.set_seed(s)


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


# ------------------------------------------------------------------ data
def load_events(run_dir, min_leads, restrict, wave_norm="shape"):
    X, Wv, meta = [], [], []
    for csv in sorted(Path(run_dir).rglob("beats_extracted.csv")):
        d = pd.read_csv(csv)
        if d.empty or "event_id" not in d.columns or "label" not in d.columns:
            continue
        npz = np.load(csv.parent / "waveforms.npz") if (csv.parent / "waveforms.npz").exists() else None
        for ev, g in d.groupby("event_id"):
            pv = g[g.beat == "PVC"]
            if pv.lead.nunique() < min_leads:
                continue
            wv = np.zeros((WAVE_LEN, 2 * NCH), np.float32)
            for j, lead in enumerate(CHANNEL_LEADS):
                r = pv[pv.lead == lead]
                if len(r) and npz is not None:
                    key = f"{lead}|PVC|c{int(r.iloc[0].column)}|p{int(r.iloc[0].pvc_index)}"
                    if key in npz.files:
                        arr = np.nan_to_num(npz[key].astype(np.float32))          # (2, T): upper, lower envelope in mV
                        T = min(WAVE_LEN, arr.shape[1])
                        wv[:T, 2 * j] = arr[0, :T]
                        wv[:T, 2 * j + 1] = arr[1, :T]
            if wave_norm == "shape":                                              # gain-independent
                wv = wv / max(0.2, float(np.abs(wv).max()))
            Wv.append(wv)
            chans, clipped, have = [], [], 0
            for lead in CHANNEL_LEADS:
                r = pv[pv.lead == lead]
                if len(r):
                    a = np.asarray(Image.open(csv.parent / "crops" / r.iloc[0].file).convert("L"), np.float32)
                    chans.append(1.0 - a / 255.0)                                # trace = 1, background = 0
                    clipped.append(bool(r.iloc[0].clipped))
                    have += 1
                else:
                    chans.append(np.zeros((CROP_H, CROP_W), np.float32))
            X.append(np.stack(chans, -1))
            meta.append({"scan_id": str(d.scan_id.iloc[0]), "patient_id": str(d.patient_id.iloc[0]),
                         "label": d.label.iloc[0], "event_id": int(ev), "n_channels": have,
                         "clipped": bool(np.any(clipped)) if clipped else False})
    meta = pd.DataFrame(meta)
    if meta.empty:
        raise SystemExit("no events found in --run-dir (run 09/10 with the current 08 first)")
    X, Wv = np.stack(X), np.stack(Wv)
    keep = meta.label.isin(["LVOT", "RVOT"]).to_numpy(copy=True)
    if restrict:
        pats = set(pd.read_csv(restrict).patient_id.astype(str))
        keep = keep & meta.patient_id.isin(pats).to_numpy()
    return X[keep], Wv[keep], meta[keep].reset_index(drop=True)


def augment(X, rng):
    """physically plausible: +-3 px (~40 ms) time shift, +-10% amplitude scale, light noise (no rotation/flip)"""
    out = np.empty_like(X)
    for i, x in enumerate(X):
        sh = int(rng.integers(-3, 4))
        a = np.roll(x, sh, axis=1)
        if sh > 0:
            a[:, :sh] = 0
        elif sh < 0:
            a[:, sh:] = 0
        sc = float(rng.uniform(0.9, 1.1))
        a = ndi.affine_transform(a, [1.0 / sc, 1.0, 1.0], offset=[(CROP_H / 2) * (1 - 1.0 / sc), 0, 0], order=1)
        out[i] = np.clip(a + rng.normal(0, 0.02, a.shape), 0, 1)
    return out


def augment_wave(W, rng):
    """shift of up to +-9 samples (~40 ms), independent amplitude scale +-10 % per lead, noise"""
    out = np.empty_like(W)
    for i, w in enumerate(W):
        sh = int(rng.integers(-9, 10))
        a = np.roll(w, sh, axis=0)
        if sh > 0:
            a[:sh] = 0
        elif sh < 0:
            a[sh:] = 0
        for j in range(0, a.shape[1], 2):
            a[:, j:j + 2] *= float(rng.uniform(0.9, 1.1))
        out[i] = a + rng.normal(0, 0.01, a.shape)
    return out


def sample_weights(y, pat):
    cnt = pd.Series(pat).map(pd.Series(pat).value_counts()).values
    w = 1.0 / np.sqrt(cnt)
    for c in (0, 1):
        w[y == c] *= 0.5 / w[y == c].sum()
    return (w * len(w) / w.sum()).astype("float32")


# ------------------------------------------------------------------ models
def prep(model_name, X):
    if model_name in ("custom", "wave1d"):
        return X
    Xu = np.repeat(np.repeat(X, MB_SCALE, axis=1), MB_SCALE, axis=2)
    return Xu * 2.0 - 1.0


def opt(lr):
    try:
        return tf.keras.optimizers.AdamW(learning_rate=lr, weight_decay=1e-4)
    except AttributeError:
        return tf.keras.optimizers.Adam(learning_rate=lr)


def compile_(model, lr):
    model.compile(optimizer=opt(lr), loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.05))


def build_custom():
    inp = layers.Input((CROP_H, CROP_W, NCH))
    x = layers.Conv2D(16, (3, 9), padding="same", activation="relu")(inp)
    x = layers.MaxPool2D(2)(x)
    x = layers.Conv2D(32, (3, 3), padding="same", activation="relu")(x)
    x = layers.MaxPool2D(2)(x)
    x = layers.Conv2D(64, (3, 3), padding="same", activation="relu", name="last_conv")(x)
    x = layers.MaxPool2D(2)(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)
    out = layers.Dense(1, activation="sigmoid", kernel_regularizer=regularizers.l2(1e-3))(x)
    return Model(inp, out), None


def build_wave1d():
    inp = layers.Input((WAVE_LEN, 2 * NCH))
    x = layers.Conv1D(16, 9, padding="same", activation="relu")(inp)
    x = layers.MaxPool1D(2)(x)
    x = layers.Conv1D(32, 5, padding="same", activation="relu")(x)
    x = layers.MaxPool1D(2)(x)
    x = layers.Conv1D(32, 3, padding="same", activation="relu", name="last_conv")(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.4)(x)
    out = layers.Dense(1, activation="sigmoid", kernel_regularizer=regularizers.l2(1e-3))(x)
    return Model(inp, out), None


MB_WEIGHTS_FILE = "mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224_no_top.h5"
MB_WEIGHTS_PATH = None


def _find_mobilenet_weights():
    cands = []
    if MB_WEIGHTS_PATH:
        cands.append(Path(MB_WEIGHTS_PATH))
    cands.append(Path.home() / ".keras" / "models" / MB_WEIGHTS_FILE)
    cands.append(Path(__file__).resolve().parent / MB_WEIGHTS_FILE)
    for c in cands:
        if c.is_file():
            return c
    return None


def _imagenet_base3(shape3):
    base3 = MobileNetV2(weights=None, include_top=False, input_shape=shape3)
    local = _find_mobilenet_weights()
    if local is not None:
        base3.load_weights(str(local))
        return base3
    try:
        return MobileNetV2(weights="imagenet", include_top=False, input_shape=shape3)
    except Exception as e:
        raise SystemExit(
            "\nCould not download the MobileNetV2 ImageNet weights (%s).\n"
            "Download this file manually:\n"
            "  https://storage.googleapis.com/tensorflow/keras-applications/mobilenet_v2/%s\n"
            "then put it in  %s  or pass  --mobilenet-weights <path-to-file>.\n"
            "(Or run with --no-imagenet for random init - NOT the same experiment.)"
            % (e, MB_WEIGHTS_FILE, Path.home() / ".keras" / "models"))


def transfer_to_nch(base3, base_n):
    """copy ImageNet weights into the N-channel MobileNetV2; only the first convolution (Conv1) is adapted:
    kernel (3,3,3,32) -> mean over RGB, replicated N times and scaled by 3/N (keeps the activation scale)"""
    for l3, ln in zip(base3.layers, base_n.layers):
        w = l3.get_weights()
        if not w:
            continue
        if l3.name == "Conv1" and w[0].shape[2] != NCH:
            k = np.repeat(w[0].mean(axis=2, keepdims=True), NCH, axis=2) * (3.0 / NCH)
            ln.set_weights([k] + w[1:])
        else:
            ln.set_weights(w)


def build_mobilenet(imagenet):
    H, W = CROP_H * MB_SCALE, CROP_W * MB_SCALE
    base = MobileNetV2(weights=None, include_top=False, input_shape=(H, W, NCH))
    if imagenet:
        transfer_to_nch(_imagenet_base3((H, W, 3)), base)
    base.trainable = False
    x = layers.GlobalAveragePooling2D()(base.output)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(1e-3))(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    return Model(base.input, out), base


def unfreeze(base, n=20):
    base.trainable = True
    for l in base.layers[:-n]:
        l.trainable = False
    for l in base.layers:
        if isinstance(l, layers.BatchNormalization):
            l.trainable = False


def train_one(name, Xtr, ytr, ptr, Xte, a, seed, imagenet):
    tf.keras.backend.clear_session()
    set_seed(seed)
    rng = np.random.default_rng(seed)
    sw = sample_weights(ytr, ptr)
    aug = augment_wave if name == "wave1d" else augment
    if name in ("custom", "wave1d"):
        model, base = build_custom() if name == "custom" else build_wave1d()
        schedule = [(a.epochs_custom, 1e-3)]
    else:
        model, base = build_mobilenet(imagenet)
        schedule = [(a.epochs_head, 1e-3), (a.epochs_ft, 1e-5)]
    snaps, total, done = [], sum(e for e, _ in schedule), 0
    for phase, (n_ep, lr) in enumerate(schedule):
        if phase == 1:
            unfreeze(base)
        compile_(model, lr)
        for _ in range(n_ep):
            model.fit(prep(name, aug(Xtr, rng)), ytr, sample_weight=sw, batch_size=16, epochs=1, shuffle=True,
                      verbose=0)
            done += 1
            if done > total - SNAP:
                snaps.append(model.predict(prep(name, Xte), verbose=0).ravel())
    return np.mean(snaps, 0), model


def train_predict(name, Xtr, ytr, ptr, Xte, a, seed, imagenet):
    preds, first = [], None
    for e in range(max(1, a.ensemble)):
        p, m = train_one(name, Xtr, ytr, ptr, Xte, a, seed + 7919 * e, imagenet)
        preds.append(p)
        first = first or m
    return np.mean(preds, 0), first


# ------------------------------------------------------------------ XAI
def _norm(c):
    return c / np.maximum(c.max(axis=tuple(range(1, c.ndim)), keepdims=True), 1e-9)


def gradcam_pair(model, layer_name, x):
    """signed Grad-CAM of the sigmoid output: raw > 0 supports RVOT, raw < 0 supports LVOT.
    returns (cam_RVOT, cam_LVOT), each ReLU-ed and normalised to 0-1 per event"""
    gm = Model(model.inputs, [model.get_layer(layer_name).output, model.output])
    x = tf.convert_to_tensor(x)
    with tf.GradientTape() as tape:
        conv, pred = gm(x)
        score = pred[:, 0]
    g = tape.gradient(score, conv)
    w = tf.reduce_mean(g, axis=tuple(range(1, len(g.shape) - 1)), keepdims=True)
    raw = tf.reduce_sum(w * conv, -1).numpy()
    return _norm(np.maximum(raw, 0)), _norm(np.maximum(-raw, 0))


def time_axis_ms_wave(n_cols):
    return (np.arange(n_cols) + 0.5) * (WAVE_LEN / n_cols) * DT_MS - PRE_MS


def time_axis_ms(n_cols):
    px = (np.arange(n_cols) + 0.5) * (CROP_W * 3.0 / n_cols)
    return (px - CANVAS_START_PX) / PX_PER_MS - PRE_MS


def lead_occlusion(name, model, Xte):
    base_p = model.predict(prep(name, Xte), verbose=0).ravel()
    out = []
    for c in range(NCH):
        Xo = Xte.copy()
        if name == "wave1d":
            Xo[..., 2 * c:2 * c + 2] = 0.0
        else:
            Xo[..., c] = 0.0
        out.append(np.abs(model.predict(prep(name, Xo), verbose=0).ravel() - base_p))
    return np.stack(out, 1)


# ------------------------------------------------------------------ statistics
def boot_ci(y, p, n=2000, seed=0, p2=None):
    rng = np.random.default_rng(seed)
    a, d = [], []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) < 2:
            continue
        a.append(roc_auc_score(y[i], p[i]))
        if p2 is not None:
            d.append(roc_auc_score(y[i], p[i]) - roc_auc_score(y[i], p2[i]))
    return tuple(np.percentile(a, [2.5, 97.5])), (tuple(np.percentile(d, [2.5, 97.5])) if d else None)


def cp_ci(k, n, alpha=0.05):
    """exact Clopper-Pearson interval"""
    if n == 0:
        return (np.nan, np.nan)
    lo = 0.0 if k == 0 else stats.beta.ppf(alpha / 2, k, n - k + 1)
    hi = 1.0 if k == n else stats.beta.ppf(1 - alpha / 2, k + 1, n - k)
    return float(lo), float(hi)


def threshold_metrics(y, p, thr):
    h = (p >= thr).astype(int)
    tn, fp, fn, tp = [int(v) for v in confusion_matrix(y, h, labels=[0, 1]).ravel()]
    sens, spec = tp / max(tp + fn, 1), tn / max(tn + fp, 1)
    ppv = tp / (tp + fp) if tp + fp else np.nan
    npv = tn / (tn + fn) if tn + fn else np.nan
    den = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    s_lo, s_hi = cp_ci(tp, tp + fn)
    p_lo, p_hi = cp_ci(tn, tn + fp)
    v_lo, v_hi = cp_ci(tp, tp + fp)
    n_lo, n_hi = cp_ci(tn, tn + fn)
    return {"threshold": thr, "TN": tn, "FP": fp, "FN": fn, "TP": tp,
            "Sens(RVOT)": sens, "Sens_lo": s_lo, "Sens_hi": s_hi,
            "Spec(LVOT)": spec, "Spec_lo": p_lo, "Spec_hi": p_hi,
            "PPV": ppv, "PPV_lo": v_lo, "PPV_hi": v_hi, "NPV": npv, "NPV_lo": n_lo, "NPV_hi": n_hi,
            "Accuracy": (tp + tn) / len(y), "BalAcc": (sens + spec) / 2,
            "F1": 2 * tp / max(2 * tp + fp + fn, 1), "MCC": (tp * tn - fp * fn) / den if den > 0 else np.nan,
            "LR+": sens / (1 - spec) if spec < 1 else np.inf, "LR-": (1 - sens) / spec if spec > 0 else np.inf}


def brier(y, p):
    return float(np.mean((p - y) ** 2))


def calibration_slope_intercept(y, p):
    """slope: logistic regression of y on logit(p); intercept (calibration-in-the-large): offset model with slope 1"""
    z = logit(p)
    if np.std(z) < 1e-6 or len(np.unique(y)) < 2:
        return np.nan, np.nan
    slope = float(LogisticRegression(C=1e6, max_iter=2000).fit(z[:, None], y).coef_[0, 0])
    try:
        inter = float(brentq(lambda s: np.mean(expit(z + s)) - np.mean(y), -12, 12))
    except ValueError:
        inter = np.nan
    return slope, inter


def calibration_bins(y, p, n_bins):
    """quantile bins; returns DataFrame with mean prediction, observed RVOT fraction and exact 95 % CI"""
    edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 3:
        edges = np.array([p.min() - 1e-9, p.max() + 1e-9])
    idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, len(edges) - 2)
    rows = []
    for b in range(len(edges) - 1):
        m = idx == b
        if not m.any():
            continue
        k, n = int(y[m].sum()), int(m.sum())
        lo, hi = cp_ci(k, n)
        rows.append({"bin": b + 1, "n": n, "mean_pred": float(p[m].mean()), "obs_frac": k / n, "ci_lo": lo, "ci_hi": hi})
    return pd.DataFrame(rows)


def ece(tab):
    return float(np.sum(tab.n * np.abs(tab.obs_frac - tab.mean_pred)) / tab.n.sum())


def platt_cv(y, p, seed, repeats=10, folds=5):
    """cross-validated Platt scaling of the OOF logit (2 parameters) -> recalibrated OOF probabilities"""
    folds = int(max(2, min(folds, np.bincount(y).min())))
    z = logit(p)[:, None]
    oo = []
    for r in range(repeats):
        o = np.full(len(y), np.nan)
        for tr, te in StratifiedKFold(folds, shuffle=True, random_state=seed + r).split(z, y):
            o[te] = LogisticRegression(C=1e4, max_iter=2000).fit(z[tr], y[tr]).predict_proba(z[te])[:, 1]
        oo.append(o)
    return np.mean(oo, 0)


def net_benefit(y, p, thr):
    n = len(y)
    out = np.empty(len(thr))
    for i, t in enumerate(thr):
        h = p >= t
        out[i] = np.sum(h & (y == 1)) / n - np.sum(h & (y == 0)) / n * t / (1 - t)
    return out


def treat_all(prev, thr):
    return prev - (1 - prev) * thr / (1 - thr)


def dca_table(y, p, thr, n_boot, seed):
    nb = net_benefit(y, p, thr)
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        bs.append(net_benefit(y[i], p[i], thr))
    lo, hi = np.percentile(bs, [2.5, 97.5], axis=0)
    return nb, lo, hi


# ------------------------------------------------------------------ evaluation block
def evaluate_scores(oofs, yp, a, out):
    """everything that is computed from the out-of-fold patient scores"""
    thr_list = sorted(set([float(a.thr)] + [float(t) for t in a.thresholds.split(",") if t.strip()]))
    prev = float(yp.mean())
    bref = prev * (1 - prev)
    recal = {n: platt_cv(yp, p, a.seed) for n, p in oofs.items()}
    rows, thr_rows, cal_rows = [], [], []
    rng = np.random.default_rng(a.seed + 1)
    for name, p in oofs.items():
        pr = recal[name]
        r = {"model": name}
        for tag, q in (("raw", p), ("recal", pr)):
            b = brier(yp, q)
            bb = []
            for _ in range(1000):
                i = rng.integers(0, len(yp), len(yp))
                bb.append(brier(yp[i], q[i]))
            sl, ic = calibration_slope_intercept(yp, q) if tag == "raw" else (np.nan, np.nan)   # slope of CV-recalibrated scores is not interpretable
            tab = calibration_bins(yp, q, a.cal_bins)
            r.update({f"Brier_{tag}": b, f"Brier_{tag}_lo": float(np.percentile(bb, 2.5)),
                      f"Brier_{tag}_hi": float(np.percentile(bb, 97.5)),
                      f"BrierSkill_{tag}": 1 - b / bref,
                      f"LogLoss_{tag}": float(log_loss(yp, np.clip(q, 1e-6, 1 - 1e-6))),
                      f"CalSlope_{tag}": sl, f"CalIntercept_{tag}": ic, f"ECE_{tag}": ece(tab)})
            for _, t in tab.iterrows():
                cal_rows.append({"model": name, "scores": tag, **t.to_dict()})
        rows.append(r)
        for t in thr_list:
            thr_rows.append({"model": name, "scores": "raw", **threshold_metrics(yp, p, t)})
        thr_rows.append({"model": name, "scores": "raw (Youden on OOF - exploratory, optimistic)",
                         **threshold_metrics(yp, p, float(roc_curve(yp, p)[2][np.argmax(
                             roc_curve(yp, p)[1] - roc_curve(yp, p)[0])]))})
    ev = pd.DataFrame(rows)
    thr_df = pd.DataFrame(thr_rows)
    cal_df = pd.DataFrame(cal_rows)
    ev.to_csv(out / "calibration_brier.csv", index=False)
    thr_df.to_csv(out / "threshold_metrics.csv", index=False)
    cal_df.to_csv(out / "calibration_bins.csv", index=False)

    # ---- DCA
    tg = np.arange(a.dca_min, a.dca_max + 1e-9, a.dca_step)
    dca_rows = []
    for tag, src in (("raw", oofs), ("recal", recal)):
        for name, q in src.items():
            nb, lo, hi = dca_table(yp, q, tg, a.dca_boot, a.seed)
            for t, v, l, h in zip(tg, nb, lo, hi):
                dca_rows.append({"scores": tag, "model": name, "threshold": float(t), "NB": float(v), "NB_lo": float(l),
                                 "NB_hi": float(h), "NB_treat_all": float(treat_all(prev, t)), "NB_treat_none": 0.0})
    dca = pd.DataFrame(dca_rows)
    dca.to_csv(out / "dca_values.csv", index=False)

    # ---- figures
    names = list(oofs)
    fig, axes = plt.subplots(1, len(names), figsize=(3.6 * len(names), 3.4), squeeze=False)
    for ax, name in zip(axes[0], names):
        cm = np.array([[v for v in (threshold_metrics(yp, oofs[name], a.thr)[k] for k in ("TN", "FP"))],
                       [v for v in (threshold_metrics(yp, oofs[name], a.thr)[k] for k in ("FN", "TP"))]])
        ax.imshow(cm, cmap="Blues", vmin=0, vmax=cm.sum(1).max())
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]}\n({100 * cm[i, j] / max(cm[i].sum(), 1):.0f}%)", ha="center", va="center",
                        color="white" if cm[i, j] > 0.6 * cm.sum(1).max() else "black")
        ax.set_xticks([0, 1], ["LVOT", "RVOT"])
        ax.set_yticks([0, 1], ["LVOT", "RVOT"])
        ax.set_xlabel("predicted")
        ax.set_ylabel("actual")
        ax.set_title(f"{name} (thr {a.thr:g})", fontsize=9)
    plt.tight_layout()
    plt.savefig(out / "confusion_matrices.png", dpi=140)
    plt.close()

    fig = plt.figure(figsize=(3.8 * len(names), 5.2))
    gs = fig.add_gridspec(2, len(names), height_ratios=[3, 1])
    for k, name in enumerate(names):
        ax = fig.add_subplot(gs[0, k])
        ax.plot([0, 1], [0, 1], ":", color="gray")
        for tag, col, ls, mk in (("raw", "tab:blue", "-", "o"), ("recal", "tab:green", "--", "s")):
            t = cal_df[(cal_df.model == name) & (cal_df.scores == tag)]
            ax.errorbar(t.mean_pred, t.obs_frac, yerr=[t.obs_frac - t.ci_lo, t.ci_hi - t.obs_frac], color=col, ls=ls,
                        marker=mk, ms=4, capsize=2, lw=1, label=tag)
        r = ev[ev.model == name].iloc[0]
        ax.set_title(f"{name}\nBrier {r.Brier_raw:.3f} (recal {r.Brier_recal:.3f}); slope {r.CalSlope_raw:.2f}", fontsize=8)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel("mean predicted P(RVOT)")
        ax.set_ylabel("observed fraction RVOT")
        ax.legend(fontsize=7, loc="upper left")
        ax2 = fig.add_subplot(gs[1, k])
        ax2.hist([oofs[name][yp == 0], oofs[name][yp == 1]], bins=np.linspace(0, 1, 11), color=["tab:blue", "tab:red"],
                 label=["LVOT", "RVOT"], stacked=True)
        ax2.set_xlim(0, 1)
        ax2.set_xlabel("predicted P(RVOT) (raw)")
        ax2.set_ylabel("patients")
        if k == 0:
            ax2.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(out / "calibration_curves.png", dpi=140)
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, tag, ttl in zip(axes, ("raw", "recal"), ("raw scores (balanced-prior training)", "recalibrated (CV Platt)")):
        d0 = dca[dca.scores == tag]
        for name in names:
            d = d0[d0.model == name]
            ax.plot(d.threshold, d.NB, label=name)
        d = d0[d0.model == names[0]]
        ax.plot(d.threshold, d.NB_treat_all, "k--", lw=1, label="treat all (as RVOT)")
        ax.axhline(0, color="gray", lw=1, label="treat none")
        ax.set_xlabel("threshold probability")
        ax.set_title(ttl, fontsize=9)
        ax.set_ylim(-0.1, max(0.05, prev + 0.05))
    axes[0].set_ylabel("net benefit")
    axes[0].legend(fontsize=7)
    plt.suptitle("Decision curve analysis (positive = RVOT; patient-level out-of-fold)", fontsize=10)
    plt.tight_layout()
    plt.savefig(out / "decision_curves.png", dpi=140)
    plt.close()
    return ev, thr_df, cal_df, dca, recal, thr_list


# ------------------------------------------------------------------ main pipeline
def pipeline(a, X, Wv, meta, out):
    out.mkdir(parents=True, exist_ok=True)
    DATA = {"custom": X, "mobilenet": X, "wave1d": Wv}
    imagenet = not a.no_imagenet
    pats = sorted(meta.patient_id.unique())
    lab = meta.groupby("patient_id").label.first().reindex(pats)
    yp = (lab == "RVOT").astype(int).values
    print(f"leads {CHANNEL_LEADS} ({NCH} channels); events {len(meta)}, patients {len(pats)} (LVOT {int((yp == 0).sum())}, "
          f"RVOT {int((yp == 1).sum())}); events/patient {meta.groupby('patient_id').size().value_counts().sort_index().to_dict()}; "
          f"events with a blank channel: {int((meta.n_channels < NCH).sum())}")
    if min(np.bincount(yp)) < 8:
        print("WARNING: fewer than 8 patients in a class - results will be extremely unstable")
    if a.folds > min(np.bincount(yp)):
        a.folds = int(min(np.bincount(yp)))
        print(f"WARNING: too few patients in a class, folds reduced to {a.folds}")
    pat_idx = {p: i for i, p in enumerate(pats)}
    ev_pat = meta.patient_id.map(pat_idx).values
    ev_y = yp[ev_pat]

    results, oofs, xai = [], {}, {}
    for name in [m.strip() for m in a.models.split(",")]:
        oo = np.full((a.repeats, len(pats)), np.nan)
        cr, cl, occl, cam_y = [], [], [], []
        for r in range(a.repeats):
            skf = StratifiedKFold(a.folds, shuffle=True, random_state=a.seed + r)
            for k, (tr_p, te_p) in enumerate(skf.split(np.zeros(len(pats)), yp)):
                tr, te = np.isin(ev_pat, tr_p), np.isin(ev_pat, te_p)
                assert set(ev_pat[tr]).isdisjoint(set(ev_pat[te])), "patient leakage"
                D = DATA[name]
                p_ev, model = train_predict(name, D[tr], ev_y[tr], ev_pat[tr], D[te], a, a.seed + 100 * r + k, imagenet)
                df = pd.DataFrame({"p": ev_pat[te], "s": p_ev}).groupby("p").s.mean()
                oo[r, df.index.values] = df.values
                if r == 0:
                    layer = "last_conv" if name in ("custom", "wave1d") else a.cam_layer_mobilenet
                    c_r, c_l = gradcam_pair(model, layer, prep(name, D[te]))
                    if c_r.ndim == 3:                                           # image CAM: average over amplitude axis
                        c_r, c_l = c_r.mean(1), c_l.mean(1)
                    cr.append(c_r)
                    cl.append(c_l)
                    occl.append(lead_occlusion(name, model, D[te]))
                    cam_y.append(ev_y[te])
                print(f"  [{name}] repeat {r + 1}/{a.repeats} fold {k + 1}/{a.folds} done", flush=True)
        mo = np.nanmean(oo, 0)
        aucs = [roc_auc_score(yp, o) for o in oo]
        ci, _ = boot_ci(yp, mo, seed=a.seed)
        tm = threshold_metrics(yp, mo, a.thr)
        results.append({"model": name, "n_patients": len(pats), "AUC": roc_auc_score(yp, mo),
                        "AUC_sd_over_repeats": float(np.std(aucs, ddof=1)) if len(aucs) > 1 else np.nan,
                        "CI_lo": ci[0], "CI_hi": ci[1], f"BalAcc@{a.thr:g}": tm["BalAcc"],
                        f"Sens(RVOT)@{a.thr:g}": tm["Sens(RVOT)"], f"Spec(LVOT)@{a.thr:g}": tm["Spec(LVOT)"]})
        oofs[name] = mo
        xai[name] = (np.concatenate(cr), np.concatenate(cl), np.concatenate(occl), np.concatenate(cam_y))
        print(f"== {name}: AUC {results[-1]['AUC']:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]  BalAcc {tm['BalAcc']:.3f}")

    res = pd.DataFrame(results)
    ev, thr_df, cal_df, dca, recal, thr_list = evaluate_scores(oofs, yp, a, out)
    res = res.merge(ev, on="model")
    res.to_csv(out / "deep12_results.csv", index=False)
    pd.DataFrame({"patient_id": pats, "y_RVOT": yp, **oofs, **{f"{k}_recal": v for k, v in recal.items()}}
                 ).to_csv(out / "deep12_oof_predictions.csv", index=False)

    # ---- paired comparison with step 13
    lines = []
    if a.compare:
        c = pd.read_csv(a.compare)
        c["patient_id"] = c.patient_id.astype(str)
        m = pd.DataFrame({"patient_id": pats, "y": yp, **oofs}).merge(c.drop(columns=[x for x in ["y_RVOT"] if x in c.columns]),
                                                                       on="patient_id")
        for ref in [x for x in c.columns if x.startswith("rule_") or "PRIMARY" in x]:
            ok = m[ref].notna().values
            if ok.sum() < a.min_compare or len(np.unique(m.y.values[ok])) < 2:
                continue
            for name in oofs:
                _, dci = boot_ci(m.y.values[ok], m[name].values[ok], seed=a.seed, p2=m[ref].values[ok])
                d = roc_auc_score(m.y.values[ok], m[name].values[ok]) - roc_auc_score(m.y.values[ok], m[ref].values[ok])
                lines.append(f"{name:10s} vs {ref:32s} n={int(ok.sum())}  AUC diff {d:+.3f}  95% CI [{dci[0]:+.3f}, {dci[1]:+.3f}]")
        print("\n".join(lines))

    # ---- ROC
    plt.figure(figsize=(5.5, 5))
    for name, mo in oofs.items():
        f, t, _ = roc_curve(yp, mo)
        plt.plot(f, t, label=f"{name} ({roc_auc_score(yp, mo):.2f})")
    plt.plot([0, 1], [0, 1], ":", color="gray")
    plt.xlabel("1 - specificity")
    plt.ylabel("sensitivity (RVOT)")
    plt.legend(loc="lower right")
    plt.title("Deep models, patient-level out-of-fold ROC")
    plt.tight_layout()
    plt.savefig(out / "deep_roc.png", dpi=140)
    plt.close()

    # ---- Grad-CAM (own-class, signed) + lead occlusion
    fig, axes = plt.subplots(1, len(xai), figsize=(5.2 * len(xai), 3.6), squeeze=False)
    occ_rows = []
    for ax, (name, (c_r, c_l, occ, cy)) in zip(axes[0], xai.items()):
        t = time_axis_ms_wave(c_r.shape[1]) if name == "wave1d" else time_axis_ms(c_r.shape[1])
        for c, col, nm in ((0, "tab:blue", "LVOT"), (1, "tab:red", "RVOT")):
            if (cy == c).any():
                prof = (c_l if c == 0 else c_r)[cy == c].mean(0)
                ax.plot(t, prof, color=col, label=f"{nm} events -> evidence for {nm} (n={int((cy == c).sum())})")
        ax.axvline(0, color="k", lw=0.8, ls="--")
        ax.set_xlabel("ms relative to the start of the annotated PVC window")
        ax.set_ylabel("mean Grad-CAM toward own class (0-1)")
        ax.set_title(f"{name}: where the evidence is (out-of-fold)", fontsize=9)
        ax.legend(fontsize=7)
        for j, ld in enumerate(CHANNEL_LEADS):
            occ_rows.append({"model": name, "lead": ld, "mean_abs_prob_change": float(occ[:, j].mean())})
    plt.tight_layout()
    plt.savefig(out / "gradcam_time_profile.png", dpi=140)
    plt.close()
    occ_df = pd.DataFrame(occ_rows)
    occ_df.to_csv(out / "lead_occlusion.csv", index=False)
    plt.figure(figsize=(max(6, 0.7 * NCH + 2), 3.4))
    w = 0.8 / max(len(xai), 1)
    for i, name in enumerate(xai):
        s = occ_df[occ_df.model == name]
        plt.bar(np.arange(NCH) + i * w, s.mean_abs_prob_change, w, label=name)
    plt.xticks(np.arange(NCH) + 0.4 - w / 2, CHANNEL_LEADS)
    plt.ylabel("mean |change in probability|\nwhen the lead is blanked")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "lead_occlusion.png", dpi=140)
    plt.close()

    # ---- report
    pd.set_option("display.width", 250)
    prim = thr_df[(thr_df.scores == "raw") & (thr_df.threshold == float(a.thr))]
    cols_thr = ["model", "TN", "FP", "FN", "TP", "Sens(RVOT)", "Spec(LVOT)", "PPV", "NPV", "Accuracy", "BalAcc", "F1", "MCC"]
    rep = ["# Deep 12-lead patient-level report\n",
           f"leads {CHANNEL_LEADS}; patients {len(pats)} (LVOT {int((yp == 0).sum())}, RVOT {int((yp == 1).sum())}, "
           f"prevalence RVOT {yp.mean():.2f}), events {len(meta)}; CV {a.repeats} x {a.folds}-fold over patients; "
           f"ensemble {a.ensemble} seed(s); wave-norm {a.wave_norm}; ImageNet weights: {imagenet}; seed {a.seed}\n",
           "\n## Discrimination (AUC) and headline metrics\n```\n" +
           res[[c for c in res.columns if not c.startswith(("Brier", "LogLoss", "Cal", "ECE"))]].round(3).to_string(index=False) + "\n```\n",
           f"\n## Confusion matrix and metrics at the fixed threshold {a.thr:g} (Clopper-Pearson CIs in threshold_metrics.csv)\n```\n" +
           prim[cols_thr].round(3).to_string(index=False) + "\n```\n",
           f"\nAll thresholds {thr_list}: see threshold_metrics.csv. The 'Youden' rows are exploratory (the threshold is chosen on the "
           "same OOF scores) and must not be reported as performance.\n",
           "\n## Brier score, calibration\n```\n" + ev.round(3).to_string(index=False) + "\n```\n",
           "Brier skill > 0 = better than always predicting the prevalence. Calibration slope < 1 = over-confident scores. "
           "'raw' probabilities come from class-balanced training (prior 50 %), 'recal' = cross-validated Platt scaling.\n",
           "\nDecision curves: decision_curves.png / dca_values.csv (net benefit with bootstrap CI). A model is only useful "
           "at thresholds where its curve lies above both 'treat all' and 'treat none'; with this n the curves are very noisy.\n"]
    if lines:
        rep.append("\n## Paired AUC differences vs step 13 (same patients, patient-bootstrap CI)\n```\n" + "\n".join(lines) + "\n```\n")
    rep.append("\nRead with care: the CI of an AUC with ~54 patients is about +-0.12; calibration/DCA from this n are descriptive. "
               "Grad-CAM and lead occlusion describe the model, not which ECG feature is causal. Run --leads V1,V2,V3 as the "
               "matching control and report every model you ran.\n")
    (out / "deep12_report.md").write_text("".join(rep), encoding="utf-8")
    print("\n" + prim[cols_thr].round(3).to_string(index=False))
    print(ev[["model", "Brier_raw", "Brier_recal", "BrierSkill_raw", "BrierSkill_recal", "CalSlope_raw", "CalIntercept_raw",
              "ECE_raw", "ECE_recal"]].round(3).to_string(index=False))
    print(f"\nreport: {(out / 'deep12_report.md').resolve()}")
    return res


# ------------------------------------------------------------------ synthetic data for --selftest
def make_synthetic_run(root, planted, n=54, n_lvot=21, seed=1):
    rng = np.random.default_rng(seed)
    root = Path(root)
    ids = []
    for i in range(n):
        lab = "LVOT" if i < n_lvot else "RVOT"
        d = root / f"scan{i}"
        (d / "crops").mkdir(parents=True, exist_ok=True)
        recs, wf = [], {}
        for e in range(int(rng.integers(1, 3))):
            for lead in ALL_LEADS:
                amp = 0.4 + rng.normal(0, 0.12) + (0.5 if (planted and lab == "RVOT" and lead in ("V1", "V2", "V3")) else 0.0)
                T = int(rng.integers(60, 95))
                t = np.arange(T)
                up = amp * np.exp(-0.5 * ((t - 14) / 5) ** 2) + rng.normal(0, 0.02, T)
                lo = -0.4 * np.exp(-0.5 * ((t - 26) / 6) ** 2)
                wf[f"{lead}|PVC|c0|p{e}"] = np.stack([up, lo]).astype(np.float32)
                img = np.full((CROP_H, CROP_W), 255, np.uint8)
                for x, c in enumerate(np.linspace(0, T - 1, CROP_W).astype(int)):
                    for v in (up[c], lo[c]):
                        img[int(np.clip(64 - v * 60, 0, CROP_H - 1)), x] = 0
                fn = f"e{e}_{lead}.png"
                Image.fromarray(img).save(d / "crops" / fn)
                recs.append({"scan_id": f"S{i}", "patient_id": f"P{i}", "label": lab, "event_id": e, "beat": "PVC",
                             "lead": lead, "column": 0, "pvc_index": e, "file": fn, "clipped": False})
        pd.DataFrame(recs).to_csv(d / "beats_extracted.csv", index=False)
        np.savez(d / "waveforms.npz", **wf)
        ids.append(f"P{i}")
    pd.DataFrame({"patient_id": ids}).to_csv(root / "patient_features.csv", index=False)


def selftest(a):
    a.epochs_custom, a.epochs_head, a.epochs_ft, a.repeats, a.no_imagenet = 25, 3, 3, 1, True
    a.dca_boot = 200
    tmp = Path(tempfile.mkdtemp(prefix="deep12_selftest_"))
    try:
        for planted in (False, True):
            root = tmp / ("planted" if planted else "null")
            make_synthetic_run(root, planted)
            print(f"\n===== SELFTEST: {'PLANTED signal (R amplitude +0.5 in V1-V3 for RVOT)' if planted else 'NULL (no signal)'}")
            X, Wv, meta = load_events(root, a.min_leads, root / "patient_features.csv", a.wave_norm)
            pipeline(a, X, Wv, meta, Path(a.out) / ("selftest_planted" if planted else "selftest_null"))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir")
    ap.add_argument("--out", default="deep12_out")
    ap.add_argument("--leads", default="all", help="'all' (12 leads) or a comma list, e.g. V1,V2,V3")
    ap.add_argument("--restrict-to", default=None, help="patient_features.csv: use exactly these patients")
    ap.add_argument("--compare", default=None, help="oof_predictions.csv of 13_model_patient_level.py")
    ap.add_argument("--min-leads", type=int, default=9)
    ap.add_argument("--min-compare", type=int, default=15)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--epochs-custom", type=int, default=40)
    ap.add_argument("--epochs-head", type=int, default=12)
    ap.add_argument("--epochs-ft", type=int, default=12)
    ap.add_argument("--models", default="wave1d,custom,mobilenet")
    ap.add_argument("--ensemble", type=int, default=1, help="seeds averaged per fold (3 recommended for the final run)")
    ap.add_argument("--wave-norm", default="shape", choices=["shape", "amplitude"])
    ap.add_argument("--thr", type=float, default=0.5, help="primary fixed decision threshold (fixed a priori; 0.5 = balanced prior)")
    ap.add_argument("--thresholds", default="0.3,0.4,0.5,0.6,0.7", help="additional fixed thresholds for threshold_metrics.csv")
    ap.add_argument("--cal-bins", type=int, default=5, help="quantile bins of the calibration curve (5 for n~54)")
    ap.add_argument("--dca-min", type=float, default=0.1)
    ap.add_argument("--dca-max", type=float, default=0.9)
    ap.add_argument("--dca-step", type=float, default=0.05)
    ap.add_argument("--dca-boot", type=int, default=500)
    ap.add_argument("--cam-layer-mobilenet", default="block_5_add")
    ap.add_argument("--no-imagenet", action="store_true")
    ap.add_argument("--mobilenet-weights", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--quick", action="store_true", help="tiny schedule for a smoke test")
    ap.add_argument("--selftest", action="store_true", help="synthetic 12-lead data: null labels -> ~0.5, planted -> high")
    a = ap.parse_args()
    if a.quick:
        a.epochs_custom, a.epochs_head, a.epochs_ft, a.repeats = 6, 3, 3, 1
    global CHANNEL_LEADS, NCH, MB_WEIGHTS_PATH
    CHANNEL_LEADS = list(ALL_LEADS) if a.leads == "all" else [s.strip() for s in a.leads.split(",") if s.strip()]
    bad = [l for l in CHANNEL_LEADS if l not in ALL_LEADS]
    if bad:
        raise SystemExit(f"unknown leads {bad}; valid: {ALL_LEADS}")
    NCH = len(CHANNEL_LEADS)
    MB_WEIGHTS_PATH = a.mobilenet_weights
    if a.selftest:
        return selftest(a)
    if not a.run_dir:
        raise SystemExit("--run-dir is required (or use --selftest)")
    X, Wv, meta = load_events(a.run_dir, a.min_leads, a.restrict_to, a.wave_norm)
    pipeline(a, X, Wv, meta, Path(a.out))


if __name__ == "__main__":
    main()
