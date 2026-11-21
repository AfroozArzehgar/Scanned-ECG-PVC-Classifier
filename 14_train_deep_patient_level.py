#!/usr/bin/env python3
"""
14_train_deep_patient_level.py
------------------------------
Deep-learning counterpart of 13_model_patient_level.py, built to keep the structure of the
paper (image input, custom CNN + transfer-learning MobileNetV2, Grad-CAM) but with a leakage-free,
physiologically sensible set-up.

INPUT   one sample = one PVC EVENT. Its 3 image channels are the scale-preserving PVC crops of
        V1, V2, V3 (the leads that carry the information; the crops are already gain-calibrated in mV
        and baseline-centred). A missing lead is replaced by a blank channel.
MODELS  (0) wave1d: a tiny 1-D CNN on the EXTRACTED mV waveforms of V1-V3 (upper+lower envelope = 6 channels) -
            no rasterisation noise, ~5k parameters (pre-specified "upgrade" of the image models)
        (1) custom CNN  (asymmetric 3x9 first kernel, 16/32/64 filters, as in the paper)
        (2) MobileNetV2 transfer learning (ImageNet weights, frozen head first, then last 20 layers
            fine-tuned with a tiny learning rate, BatchNorm frozen)
ENSEMBLE --ensemble N trains N seeds per fold and averages them (reduces the variance that dominates with ~54 patients)
TRAINING fixed schedule (no early stopping on tiny validation sets), class- and patient-balanced sample
        weights, physically plausible augmentation only (+-40 ms time shift, +-10 % amplitude, noise;
        NO rotation, NO flips), predictions averaged over the last 5 epochs (snapshot averaging)
EVAL    repeated stratified K-fold OVER PATIENTS (default 3 x 5). A patient is predicted only by models that
        never saw any of his events; the patient score = mean over his events and snapshots.
        Reported (all patient-level, patient-bootstrap 95 % CI):
          - AUC, ROC curve
          - Balanced accuracy, sensitivity (RVOT), specificity (LVOT), PPV, NPV and the CONFUSION MATRIX at a
            FIXED, pre-specified decision threshold (--threshold, default 0.5; never tuned on the OOF scores)
          - Brier score (with the "always predict the prevalence" reference), calibration curve,
            calibration slope / intercept
          - Decision Curve Analysis (net benefit vs. treat-all / treat-none)
        Every figure is written as PNG and as SVG.
XAI     Grad-CAM from the out-of-fold models (time profile of the evidence), plus LEAD OCCLUSION
        (how much the prediction changes when one lead channel is blanked).

The lead-ablation experiment (re-training with V1/V2/V3 removed) lives in 15_lead_ablation.py, which re-uses
the functions of this file.

Usage
  python 14_train_deep_patient_level.py --run-dir validation_out --out deep_out \\
        --restrict-to features_out/patient_features.csv --compare model_out2/oof_predictions.csv
  python 14_train_deep_patient_level.py --run-dir validation_out --out deep_quick --quick --no-imagenet   # smoke test
"""
import argparse
import os
import random
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
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold

import tensorflow as tf
from tensorflow.keras import layers, Model, regularizers
from tensorflow.keras.applications import MobileNetV2

plt.rcParams["svg.fonttype"] = "none"      # keep text as real text in the SVG (editable in Inkscape/Illustrator); use "path" to embed outlines

CHANNEL_LEADS = ["V1", "V2", "V3"]
CROP_H, CROP_W = 128, 64            # size of the stored crops
CANVAS_W_PX, PX_PER_MS = 192.0, 0.2174
CANVAS_START_PX, PRE_MS = 20.0, 40.0   # window starts at canvas x = 20 px; crops start 40 ms before the window
WAVE_LEN = 128                      # samples of the 1-D input (4.6 ms each, from 40 ms before the window start)
DT_MS = 1000.0 / 217.44
SNAP = 5                            # number of last epochs averaged
MB_SCALE = 2                        # MobileNetV2 gets the crops up-sampled x2 (finer Grad-CAM, >= 32 px)


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    tf.random.set_seed(s)


# ------------------------------------------------------------------ data
def load_events(run_dir, min_leads, restrict):
    """returns images X (n,128,64,3), RAW (un-normalised) waveforms Wv (n,128,6) and the event table.
    The waveform normalisation is done separately (normalize_wave) so that a lead-ablation can first
    remove a lead and only then normalise (otherwise the scale factor would leak the removed lead)."""
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
            wv = np.zeros((WAVE_LEN, 2 * len(CHANNEL_LEADS)), np.float32)
            for j, lead in enumerate(CHANNEL_LEADS):
                r = pv[pv.lead == lead]
                if len(r) and npz is not None:
                    key = f"{lead}|PVC|c{int(r.iloc[0].column)}|p{int(r.iloc[0].pvc_index)}"
                    if key in npz.files:
                        arr = np.nan_to_num(npz[key].astype(np.float32))          # (2, T): upper, lower envelope in mV
                        T = min(WAVE_LEN, arr.shape[1])
                        wv[:T, 2 * j] = arr[0, :T]
                        wv[:T, 2 * j + 1] = arr[1, :T]
            Wv.append(wv)
            chans, clipped, have = [], [], 0
            for lead in CHANNEL_LEADS:
                r = pv[pv.lead == lead]
                if len(r):
                    a = np.asarray(Image.open(csv.parent / "crops" / r.iloc[0].file).convert("L"), np.float32)
                    chans.append(1.0 - a / 255.0)                        # trace = 1, background = 0
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
    keep = meta.label.isin(["LVOT", "RVOT"]).to_numpy(copy=True)      # writable copy (recent pandas returns read-only views)
    if restrict:
        pats = set(pd.read_csv(restrict).patient_id.astype(str))
        keep = keep & meta.patient_id.isin(pats).to_numpy()
    return X[keep], Wv[keep], meta[keep].reset_index(drop=True)


def normalize_wave(Wv, wave_norm="shape"):
    """shape: gain-independent, divide each event by its largest excursion (same rule as before);
    amplitude: keep mV (needs verified gain)"""
    if wave_norm != "shape":
        return Wv
    mx = np.maximum(0.2, np.abs(Wv).reshape(len(Wv), -1).max(1))
    return (Wv / mx[:, None, None]).astype(np.float32)


def augment(X, rng):
    """physically plausible: +-3 px (~40 ms) time shift, +-10% amplitude scale, light noise"""
    out = np.empty_like(X)
    for i, x in enumerate(X):
        sh = int(rng.integers(-3, 4))
        a = np.roll(x, sh, axis=1)
        if sh > 0:
            a[:, :sh] = 0
        elif sh < 0:
            a[:, sh:] = 0
        sc = float(rng.uniform(0.9, 1.1))                                   # amplitude (vertical) scale about the centre row
        a = ndi.affine_transform(a, [1.0 / sc, 1.0, 1.0], offset=[(CROP_H / 2) * (1 - 1.0 / sc), 0, 0], order=1)
        out[i] = np.clip(a + rng.normal(0, 0.02, a.shape), 0, 1)
    return out


def augment_wave(W, rng):
    """+-3 samples (~14 ms) ... shift of up to +-9 samples (~40 ms), independent amplitude scale +-10 % per lead, noise"""
    out = np.empty_like(W)
    for i, w in enumerate(W):
        sh = int(rng.integers(-9, 10))
        a = np.roll(w, sh, axis=0)
        if sh > 0:
            a[:sh] = 0
        elif sh < 0:
            a[sh:] = 0
        for j in range(0, a.shape[1], 2):                                         # one scale per lead (upper+lower together)
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
    return Xu * 2.0 - 1.0                                                   # MobileNetV2 expects [-1, 1]


def opt(lr):
    try:
        return tf.keras.optimizers.AdamW(learning_rate=lr, weight_decay=1e-4)
    except AttributeError:
        return tf.keras.optimizers.Adam(learning_rate=lr)


def compile_(model, lr):
    model.compile(optimizer=opt(lr), loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.05))


def build_custom():
    inp = layers.Input((CROP_H, CROP_W, 3))
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
    inp = layers.Input((WAVE_LEN, 2 * len(CHANNEL_LEADS)))
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
MB_WEIGHTS_PATH = None              # set from --mobilenet-weights


def _find_mobilenet_weights():
    """local ImageNet weights file: explicit --mobilenet-weights, else the Keras cache (~/.keras/models)"""
    cands = []
    if MB_WEIGHTS_PATH:
        cands.append(Path(MB_WEIGHTS_PATH))
    cands.append(Path.home() / ".keras" / "models" / MB_WEIGHTS_FILE)
    cands.append(Path(__file__).resolve().parent / MB_WEIGHTS_FILE)
    for c in cands:
        if c.is_file():
            return c
    return None


def build_mobilenet(imagenet):
    shape = (CROP_H * MB_SCALE, CROP_W * MB_SCALE, 3)
    if not imagenet:
        base = MobileNetV2(weights=None, include_top=False, input_shape=shape)
    else:
        local = _find_mobilenet_weights()
        if local is not None:
            # build without download, then load the (shape-independent) conv weights from the local file
            base = MobileNetV2(weights=None, include_top=False, input_shape=shape)
            base.load_weights(str(local))
        else:
            try:
                base = MobileNetV2(weights="imagenet", include_top=False, input_shape=shape)
            except Exception as e:
                raise SystemExit(
                    "\nCould not download the MobileNetV2 ImageNet weights (%s).\n"
                    "Download this file manually (VPN / other machine / browser):\n"
                    "  https://storage.googleapis.com/tensorflow/keras-applications/mobilenet_v2/%s\n"
                    "then either put it in  %s  or pass  --mobilenet-weights <path-to-file>.\n"
                    "(Or run with --no-imagenet for random init - NOT the same experiment.)"
                    % (e, MB_WEIGHTS_FILE, Path.home() / ".keras" / "models"))
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
            model.fit(prep(name, aug(Xtr, rng)), ytr, sample_weight=sw, batch_size=16, epochs=1,
                      shuffle=True, verbose=0)
            done += 1
            if done > total - SNAP:
                snaps.append(model.predict(prep(name, Xte), verbose=0).ravel())
    return np.mean(snaps, 0), model


def train_predict(name, Xtr, ytr, ptr, Xte, a, seed, imagenet):
    """snapshot-averaged (last 5 epochs) and seed-averaged (--ensemble) probabilities; XAI uses the first seed's model"""
    preds, first = [], None
    for e in range(max(1, a.ensemble)):
        p, m = train_one(name, Xtr, ytr, ptr, Xte, a, seed + 7919 * e, imagenet)
        preds.append(p)
        first = first or m
    return np.mean(preds, 0), first


# ------------------------------------------------------------------ XAI
def gradcam(model, layer_name, x):
    gm = Model(model.inputs, [model.get_layer(layer_name).output, model.output])
    with tf.GradientTape() as tape:
        conv, pred = gm(x)
        score = pred[:, 0]
    g = tape.gradient(score, conv)
    w = tf.reduce_mean(g, axis=(1, 2), keepdims=True)
    cam = tf.nn.relu(tf.reduce_sum(w * conv, -1)).numpy()                    # (n, h, w)
    m = cam.reshape(len(cam), -1).max(1)[:, None, None]
    return cam / np.maximum(m, 1e-9)


def gradcam1d(model, x):
    gm = Model(model.inputs, [model.get_layer("last_conv").output, model.output])
    with tf.GradientTape() as tape:
        conv, pred = gm(x)
        score = pred[:, 0]
    g = tape.gradient(score, conv)
    w = tf.reduce_mean(g, axis=1, keepdims=True)
    cam = tf.nn.relu(tf.reduce_sum(w * conv, -1)).numpy()                     # (n, t)
    return cam / np.maximum(cam.max(1, keepdims=True), 1e-9)


def time_axis_ms_wave(n_cols):
    return (np.arange(n_cols) + 0.5) * (WAVE_LEN / n_cols) * DT_MS - PRE_MS


def time_axis_ms(n_cols):
    """centre of each CAM column in ms relative to the START of the annotated window"""
    px = (np.arange(n_cols) + 0.5) * (CROP_W * 3.0 / n_cols)                # canvas px (crop = canvas / 3)
    return (px - CANVAS_START_PX) / PX_PER_MS - PRE_MS


def lead_occlusion(name, model, Xte):
    base_p = model.predict(prep(name, Xte), verbose=0).ravel()
    out = []
    for c in range(3):
        Xo = Xte.copy()
        if name == "wave1d":
            Xo[..., 2 * c:2 * c + 2] = 0.0
        else:
            Xo[..., c] = 0.0
        out.append(np.abs(model.predict(prep(name, Xo), verbose=0).ravel() - base_p))
    return np.stack(out, 1)                                                   # (n, 3)


# ------------------------------------------------------------------ evaluation helpers
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


# ------------------------------------------------------------------ patient set-up + cross-validation (shared with 15_lead_ablation.py)
def prepare_patients(meta, folds):
    pats = sorted(meta.patient_id.unique())
    lab = meta.groupby("patient_id").label.first().reindex(pats)
    yp = (lab == "RVOT").astype(int).values
    if folds > min(np.bincount(yp)):
        folds = int(min(np.bincount(yp)))
        print(f"WARNING: too few patients in a class, folds reduced to {folds}")
    pat_idx = {p: i for i, p in enumerate(pats)}
    ev_pat = meta.patient_id.map(pat_idx).values
    return pats, yp, ev_pat, yp[ev_pat], folds


def cross_validate(name, D, yp, ev_pat, ev_y, a, imagenet, with_xai=True, label=None):
    """repeated stratified K-fold over patients. Returns oo (repeats x patients out-of-fold patient scores),
    and (only if with_xai) Grad-CAMs, lead-occlusion changes and event labels of repeat 0.
    The splits depend only on (--seed, repeat), so different input variants are evaluated on IDENTICAL splits."""
    tag = label or name
    oo = np.full((a.repeats, len(yp)), np.nan)
    cams, occl, cam_y = [], [], []
    for r in range(a.repeats):
        skf = StratifiedKFold(a.folds, shuffle=True, random_state=a.seed + r)
        for k, (tr_p, te_p) in enumerate(skf.split(np.zeros(len(yp)), yp)):
            tr = np.isin(ev_pat, tr_p)
            te = np.isin(ev_pat, te_p)
            assert set(ev_pat[tr]).isdisjoint(set(ev_pat[te])), "patient leakage"
            p_ev, model = train_predict(name, D[tr], ev_y[tr], ev_pat[tr], D[te], a,
                                        a.seed + 100 * r + k, imagenet)
            df = pd.DataFrame({"p": ev_pat[te], "s": p_ev}).groupby("p").s.mean()
            oo[r, df.index.values] = df.values
            if with_xai and r == 0:                                       # XAI only from repeat 0
                if name == "wave1d":
                    cams.append(gradcam1d(model, D[te]))
                else:
                    layer = "last_conv" if name == "custom" else a.cam_layer_mobilenet
                    cams.append(gradcam(model, layer, prep(name, D[te])))
                occl.append(lead_occlusion(name, model, D[te]))
                cam_y.append(ev_y[te])
            print(f"  [{tag}] repeat {r + 1}/{a.repeats} fold {k + 1}/{a.folds} done", flush=True)
    return oo, cams, occl, cam_y


# ------------------------------------------------------------------ metrics at a fixed threshold, calibration, DCA
def all_metrics(y, p, thr):
    """AUC (patient score), Brier, and the confusion-matrix based metrics at the FIXED threshold thr (RVOT = positive)"""
    h = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, h, labels=[0, 1]).ravel()
    sens = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    return {"AUC": roc_auc_score(y, p) if len(np.unique(y)) == 2 else np.nan,
            "BalAcc": (sens + spec) / 2, "Sens": sens, "Spec": spec,
            "PPV": tp / (tp + fp) if (tp + fp) else np.nan, "NPV": tn / (tn + fn) if (tn + fn) else np.nan,
            "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
            "Brier": brier_score_loss(y, p)}


def boot_metrics(y, p, thr, n=2000, seed=0):
    """patient bootstrap (resample PATIENTS with replacement) -> 95 % percentile CI of AUC, BalAcc, Sens, Spec, Brier"""
    rng = np.random.default_rng(seed)
    keys = ["AUC", "BalAcc", "Sens", "Spec", "Brier"]
    acc = {k: [] for k in keys}
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) < 2:
            continue
        m = all_metrics(y[i], p[i], thr)
        for k in keys:
            acc[k].append(m[k])
    return {k: tuple(np.percentile(v, [2.5, 97.5])) for k, v in acc.items()}


def calibration_slope_intercept(y, p):
    """logistic recalibration  logit P(RVOT) = intercept + slope * logit(p).  slope<1: over-confident, >1: under-confident.
    Very noisy with ~54 patients - descriptive only."""
    q = np.clip(p, 1e-4, 1 - 1e-4)
    z = np.log(q / (1 - q)).reshape(-1, 1)
    try:
        lr = LogisticRegression(C=1e6, max_iter=1000).fit(z, y)
        return float(lr.coef_[0, 0]), float(lr.intercept_[0])
    except Exception:
        return np.nan, np.nan


def summarize(yp, p, thr, n_boot, seed, aucs=None):
    """one result row: discrimination, fixed-threshold performance (+ patient-bootstrap CI), Brier, calibration"""
    m = all_metrics(yp, p, thr)
    ci = boot_metrics(yp, p, thr, n_boot, seed)
    prev = float(yp.mean())
    slope, icpt = calibration_slope_intercept(yp, p)
    return {"n_patients": len(yp), "AUC": m["AUC"],
            "AUC_sd_over_repeats": float(np.std(aucs, ddof=1)) if aucs is not None and len(aucs) > 1 else np.nan,
            "CI_lo": ci["AUC"][0], "CI_hi": ci["AUC"][1], "threshold": thr,
            "BalAcc": m["BalAcc"], "BalAcc_lo": ci["BalAcc"][0], "BalAcc_hi": ci["BalAcc"][1],
            "Sens(RVOT)": m["Sens"], "Sens_lo": ci["Sens"][0], "Sens_hi": ci["Sens"][1],
            "Spec(LVOT)": m["Spec"], "Spec_lo": ci["Spec"][0], "Spec_hi": ci["Spec"][1],
            "PPV": m["PPV"], "NPV": m["NPV"], "TN": m["TN"], "FP": m["FP"], "FN": m["FN"], "TP": m["TP"],
            "Brier": m["Brier"], "Brier_lo": ci["Brier"][0], "Brier_hi": ci["Brier"][1],
            "Brier_ref_prevalence": prev * (1 - prev), "Cal_slope": slope, "Cal_intercept": icpt}


def net_benefit(y, p, pt):
    """net benefit of 'call RVOT if p >= pt' (Vickers & Elkin): TP/n - FP/n * pt/(1-pt)"""
    n = len(y)
    h = p >= pt
    tp = float(np.sum(h & (y == 1)))
    fp = float(np.sum(h & (y == 0)))
    return tp / n - fp / n * pt / (1 - pt)


def decision_curves(oofs, yp, pts):
    prev = float(yp.mean())
    rows = []
    for pt in pts:
        rows.append({"threshold": pt, "model": "treat_all", "net_benefit": prev - (1 - prev) * pt / (1 - pt)})
        rows.append({"threshold": pt, "model": "treat_none", "net_benefit": 0.0})
        for name, p in oofs.items():
            rows.append({"threshold": pt, "model": name, "net_benefit": net_benefit(yp, p, pt)})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ figures (PNG + SVG)
def save_fig(fig, out, stem, dpi=140):
    out = Path(out)
    fig.savefig(out / f"{stem}.png", dpi=dpi)
    fig.savefig(out / f"{stem}.svg")
    plt.close(fig)


def plot_roc(oofs, yp, out):
    fig = plt.figure(figsize=(5.5, 5))
    for name, mo in oofs.items():
        f, t, _ = roc_curve(yp, mo)
        plt.plot(f, t, label=f"{name} ({roc_auc_score(yp, mo):.2f})")
    plt.plot([0, 1], [0, 1], ":", color="gray")
    plt.xlabel("1 - specificity")
    plt.ylabel("sensitivity (RVOT)")
    plt.legend(loc="lower right")
    plt.title("Deep models, patient-level out-of-fold ROC")
    plt.tight_layout()
    save_fig(fig, out, "deep_roc")


def plot_confusion(oofs, yp, thr, out):
    n = len(oofs)
    fig, axes = plt.subplots(1, n, figsize=(3.6 * n, 3.5), squeeze=False)
    for ax, (name, p) in zip(axes[0], oofs.items()):
        cm = confusion_matrix(yp, (p >= thr).astype(int), labels=[0, 1])
        frac = cm / np.maximum(cm.sum(1, keepdims=True), 1)               # row-normalised: sensitivity / specificity
        ax.imshow(frac, cmap="Blues", vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]}\n({100 * frac[i, j]:.0f}%)", ha="center", va="center",
                        color="white" if frac[i, j] > 0.5 else "black", fontsize=11)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["LVOT", "RVOT"])
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["LVOT", "RVOT"])
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        ax.set_title(f"{name}  (threshold {thr:.2f})", fontsize=10)
    fig.suptitle("Patient-level confusion matrices (counts, % of true class)", fontsize=10)
    plt.tight_layout()
    save_fig(fig, out, "confusion_matrices")


def plot_calibration(oofs, yp, nbins, strategy, out):
    n = len(oofs)
    fig = plt.figure(figsize=(4.2 * n, 5.6))
    gs = fig.add_gridspec(2, n, height_ratios=[3, 1], hspace=0.08)
    bins = np.linspace(0, 1, 11)
    for j, (name, p) in enumerate(oofs.items()):
        ax = fig.add_subplot(gs[0, j])
        axh = fig.add_subplot(gs[1, j], sharex=ax)
        try:
            frac_pos, mean_pred = calibration_curve(yp, p, n_bins=nbins, strategy=strategy)
            ax.plot(mean_pred, frac_pos, "o-", color="tab:red", label=f"{nbins} {strategy} bins")
        except ValueError:
            pass
        ax.plot([0, 1], [0, 1], ":", color="gray", label="perfect")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_ylabel("observed fraction RVOT")
        ax.set_title(f"{name}\nBrier {brier_score_loss(yp, p):.3f} (prevalence-only ref. {yp.mean() * (1 - yp.mean()):.3f})",
                     fontsize=9)
        ax.legend(loc="upper left", fontsize=8)
        plt.setp(ax.get_xticklabels(), visible=False)
        axh.hist(p[yp == 0], bins=bins, alpha=0.6, color="tab:blue", label="LVOT")
        axh.hist(p[yp == 1], bins=bins, alpha=0.6, color="tab:red", label="RVOT")
        axh.set_xlabel("predicted probability of RVOT")
        axh.set_ylabel("patients")
        if j == 0:
            axh.legend(fontsize=7)
    fig.suptitle("Calibration (patient-level out-of-fold)", fontsize=10)
    save_fig(fig, out, "calibration_curves")


def plot_dca(dca, yp, thr, out):
    fig = plt.figure(figsize=(6, 4.4))
    prev = float(yp.mean())
    for name, g in dca.groupby("model", sort=False):
        if name == "treat_all":
            plt.plot(g.threshold, g.net_benefit, color="gray", lw=1.2, label="treat all (call everyone RVOT)")
        elif name == "treat_none":
            plt.plot(g.threshold, g.net_benefit, color="k", lw=1.2, label="treat none")
        else:
            plt.plot(g.threshold, g.net_benefit, lw=1.8, label=name)
    plt.axvline(thr, color="k", ls="--", lw=0.7)
    plt.ylim(-0.1, prev * 1.1)
    plt.xlabel("threshold probability")
    plt.ylabel("net benefit")
    plt.title("Decision curve analysis (out-of-fold, patient level)")
    plt.legend(fontsize=8)
    plt.tight_layout()
    save_fig(fig, out, "decision_curve")


# ------------------------------------------------------------------ command line shared with 15_lead_ablation.py
def add_common_args(ap):
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--restrict-to", default=None, help="patient_features.csv: use exactly these patients")
    ap.add_argument("--min-leads", type=int, default=9)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--epochs-custom", type=int, default=40)
    ap.add_argument("--epochs-head", type=int, default=12)
    ap.add_argument("--epochs-ft", type=int, default=12)
    ap.add_argument("--models", default="wave1d,custom,mobilenet")
    ap.add_argument("--ensemble", type=int, default=1, help="seeds averaged per fold (3 recommended for the final run)")
    ap.add_argument("--wave-norm", default="shape", choices=["shape", "amplitude"],
                    help="shape: divide each event by its largest excursion (gain-independent); amplitude: keep mV (needs verified gain)")
    ap.add_argument("--no-imagenet", action="store_true")
    ap.add_argument("--mobilenet-weights", default=None, help="local path of the MobileNetV2 *_no_top.h5 ImageNet weights (avoids the download)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--quick", action="store_true", help="tiny schedule for a smoke test")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="FIXED decision threshold on the patient score (pre-specify it; do not tune it on the OOF scores)")
    ap.add_argument("--n-boot", type=int, default=2000, help="patient-bootstrap replicates for the confidence intervals")


def finalize_args(a):
    """apply --quick, publish the MobileNet weight path; returns the imagenet flag"""
    global MB_WEIGHTS_PATH
    if a.quick:
        a.epochs_custom, a.epochs_head, a.epochs_ft, a.repeats = 6, 3, 3, 1
    MB_WEIGHTS_PATH = a.mobilenet_weights
    return not a.no_imagenet


def main():
    ap = argparse.ArgumentParser()
    add_common_args(ap)
    ap.add_argument("--out", default="deep_out")
    ap.add_argument("--compare", default=None, help="model_out/oof_predictions.csv of 13_model_patient_level.py")
    ap.add_argument("--min-compare", type=int, default=15, help="min patients for a paired comparison")
    ap.add_argument("--cam-layer-mobilenet", default="block_5_add")
    ap.add_argument("--cal-bins", type=int, default=5, help="bins of the calibration curve (few: only ~54 patients)")
    ap.add_argument("--cal-strategy", default="quantile", choices=["quantile", "uniform"],
                    help="quantile: equal number of patients per bin; uniform: equal-width probability bins")
    a = ap.parse_args()
    imagenet = finalize_args(a)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    X, Wraw, meta = load_events(a.run_dir, a.min_leads, a.restrict_to)
    DATA = {"custom": X, "mobilenet": X, "wave1d": normalize_wave(Wraw, a.wave_norm)}
    pats, yp, ev_pat, ev_y, a.folds = prepare_patients(meta, a.folds)
    print(f"events {len(meta)}, patients {len(pats)} (LVOT {int((yp == 0).sum())}, RVOT {int((yp == 1).sum())}); "
          f"events/patient {meta.groupby('patient_id').size().value_counts().sort_index().to_dict()}; "
          f"events with a blank channel: {int((meta.n_channels < 3).sum())}")

    results, oofs, xai = [], {}, {}
    for name in [m.strip() for m in a.models.split(",")]:
        oo, cams, occl, cam_y = cross_validate(name, DATA[name], yp, ev_pat, ev_y, a, imagenet)
        mo = np.nanmean(oo, 0)
        aucs = [roc_auc_score(yp, o) for o in oo]
        row = summarize(yp, mo, a.threshold, a.n_boot, a.seed, aucs)
        results.append({"model": name, **row})
        oofs[name] = mo
        xai[name] = (np.concatenate(cams), np.concatenate(occl), np.concatenate(cam_y))
        print(f"== {name}: AUC {row['AUC']:.3f} [{row['CI_lo']:.3f}, {row['CI_hi']:.3f}]  "
              f"@thr {a.threshold:.2f}: BalAcc {row['BalAcc']:.3f}  Sens {row['Sens(RVOT)']:.3f}  Spec {row['Spec(LVOT)']:.3f}  "
              f"Brier {row['Brier']:.3f} (ref {row['Brier_ref_prevalence']:.3f})")

    res = pd.DataFrame(results)
    res.to_csv(out / "deep_results.csv", index=False)
    pd.DataFrame({"patient_id": pats, "y_RVOT": yp, **oofs}).to_csv(out / "deep_oof_predictions.csv", index=False)

    # ---- paired comparison with the models of step 13
    lines = []
    if a.compare:
        c = pd.read_csv(a.compare)
        c["patient_id"] = c.patient_id.astype(str)
        m = pd.DataFrame({"patient_id": pats, "y": yp, **oofs}).merge(c.drop(columns=["y_RVOT"]), on="patient_id")
        for ref in [x for x in c.columns if x.startswith("rule_") or "PRIMARY" in x]:
            ok = m[ref].notna().values
            if ok.sum() < a.min_compare:
                continue
            for name in oofs:
                (_, dci) = boot_ci(m.y.values[ok], m[name].values[ok], seed=a.seed, p2=m[ref].values[ok])
                d = roc_auc_score(m.y.values[ok], m[name].values[ok]) - roc_auc_score(m.y.values[ok], m[ref].values[ok])
                lines.append(f"{name:10s} vs {ref:32s} n={int(ok.sum())}  AUC diff {d:+.3f}  95% CI [{dci[0]:+.3f}, {dci[1]:+.3f}]")
        print("\n".join(lines))

    # ---- plots: ROC, confusion matrices, calibration, decision curve analysis (PNG + SVG)
    plot_roc(oofs, yp, out)
    plot_confusion(oofs, yp, a.threshold, out)
    plot_calibration(oofs, yp, a.cal_bins, a.cal_strategy, out)
    dca = decision_curves(oofs, yp, np.round(np.arange(0.05, 0.951, 0.01), 2))
    dca.to_csv(out / "decision_curve.csv", index=False)
    plot_dca(dca, yp, a.threshold, out)

    # ---- XAI: Grad-CAM time profile per class + lead occlusion
    fig, axes = plt.subplots(1, len(xai), figsize=(5.2 * len(xai), 3.6), squeeze=False)
    occ_rows = []
    for ax, (name, (cam, occ, cy)) in zip(axes[0], xai.items()):
        prof = cam if cam.ndim == 2 else cam.mean(1)                          # image CAM: average over amplitude axis -> (n, w)
        t = time_axis_ms_wave(prof.shape[1]) if name == "wave1d" else time_axis_ms(prof.shape[1])
        for c, col, nm in ((0, "tab:blue", "LVOT"), (1, "tab:red", "RVOT")):
            if (cy == c).any():
                ax.plot(t, prof[cy == c].mean(0), color=col, label=f"{nm} (n={int((cy == c).sum())} events)")
        ax.axvline(0, color="k", lw=0.8, ls="--")
        ax.set_xlabel("ms relative to the start of the annotated PVC window")
        ax.set_ylabel("mean Grad-CAM (0-1)")
        ax.set_title(f"{name}: where the evidence is (out-of-fold)")
        ax.legend(fontsize=8)
        for j, ld in enumerate(CHANNEL_LEADS):
            occ_rows.append({"model": name, "lead": ld, "mean_abs_prob_change": float(occ[:, j].mean())})
    plt.tight_layout()
    save_fig(fig, out, "gradcam_time_profile")
    occ_df = pd.DataFrame(occ_rows)
    occ_df.to_csv(out / "lead_occlusion.csv", index=False)
    fig = plt.figure(figsize=(5, 3.2))
    w = 0.38
    for i, name in enumerate(xai):
        s = occ_df[occ_df.model == name]
        plt.bar(np.arange(3) + i * w, s.mean_abs_prob_change, w, label=name)
    plt.xticks(np.arange(3) + w / 2, CHANNEL_LEADS)
    plt.ylabel("mean |change in probability|\nwhen the lead is blanked")
    plt.legend()
    plt.tight_layout()
    save_fig(fig, out, "lead_occlusion")

    disc = ["model", "n_patients", "AUC", "CI_lo", "CI_hi", "AUC_sd_over_repeats"]
    thr_cols = ["model", "threshold", "BalAcc", "BalAcc_lo", "BalAcc_hi", "Sens(RVOT)", "Sens_lo", "Sens_hi",
                "Spec(LVOT)", "Spec_lo", "Spec_hi", "PPV", "NPV", "TN", "FP", "FN", "TP"]
    cal_cols = ["model", "Brier", "Brier_lo", "Brier_hi", "Brier_ref_prevalence", "Cal_slope", "Cal_intercept"]
    rep = ["# Deep patient-level report\n",
           f"patients {len(pats)} (LVOT {int((yp == 0).sum())}, RVOT {int((yp == 1).sum())}), events {len(meta)}; "
           f"CV {a.repeats} x {a.folds}-fold over patients; ensemble {a.ensemble} seed(s); wave-norm {a.wave_norm}; "
           f"ImageNet weights: {imagenet}; seed {a.seed}; fixed threshold {a.threshold}\n",
           "\n## Discrimination (patient-bootstrap 95 % CI)\n```\n" + res[disc].round(3).to_string(index=False) + "\n```\n",
           f"\n## Performance at the fixed threshold {a.threshold} (positive = RVOT; confusion matrix: TN/FP/FN/TP)\n```\n"
           + res[thr_cols].round(3).to_string(index=False) + "\n```\n",
           "\n## Calibration\n```\n" + res[cal_cols].round(3).to_string(index=False) + "\n```\n"]
    if lines:
        rep.append("\nPaired AUC differences (same patients, patient-bootstrap CI):\n```\n" + "\n".join(lines) + "\n```\n")
    rep.append("\nRead with care: the CI of an AUC with ~54 patients is about +-0.12; Grad-CAM and lead occlusion "
               "are exploratory descriptions of the model, not proof of which ECG feature is causal.\n"
               "Calibration / DCA: the networks are trained with class-balanced sample weights and label smoothing, and the patient "
               "score is an average over snapshots, events and CV repeats, so the scores are NOT calibrated probabilities of the "
               "real prevalence. Brier score and the decision curve are therefore descriptive; a decision-curve threshold "
               "probability is only interpretable after recalibration on independent data.\n"
               "The fixed threshold is pre-specified; choosing it on these out-of-fold scores (e.g. Youden) would make "
               "BalAcc/Sens/Spec optimistic.\n")
    (out / "deep_report.md").write_text("".join(rep), encoding="utf-8")
    print(f"\nreport: {(out / 'deep_report.md').resolve()}")


if __name__ == "__main__":
    main()
