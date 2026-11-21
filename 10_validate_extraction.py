#!/usr/bin/env python3
"""
10_validate_extraction.py
-------------------------
Validate the scan -> beat extraction (08_extract_pvc_sinus_from_scan.py) on ALL
of your scans, locally, and produce a small PHI-free bundle you can inspect or
share.

What it checks
  A. Agreement with the cardiologist's coloured lines (wherever they exist):
     is the automatic PVC / sinus classification the same, and how far (ms) is
     the automatic window onset from the cardiologist's onset?
  B. Held-out calibration: onset offsets are estimated on a random subset of
     scans (calibration set) and evaluated on the rest (test set).
  C. Physiological plausibility on EVERY column (also those without lines):
     is a detected PVC premature (coupling < 90 % of the preceding sinus RR) and
     wider than the sinus beats?
  D. Panel reliability per lead, deskew angle, clipping, PVC events per scan.
  E. A review list (scans to look at manually) and a random audit sheet with a
     Wilson-CI summary of YOUR manual review (--audit-summary).

Usage
  python 10_validate_extraction.py --scans scans.csv --out validation_out
  python 10_validate_extraction.py --scans scans.csv --out validation_out --force
  python 10_validate_extraction.py --audit-summary validation_out/audit_sheet.csv

scans.xlsx (made by 11_make_scans_xlsx.py) or scans.csv with columns:
  file, patient_id, gain_mV_per_cm  (+ scan_id, label, file_suffix, gain_verified)
"""
import argparse
import hashlib
import importlib.util
import json
import math
import shutil
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------
# EDIT HERE if you press "Run" in VS Code (no terminal arguments).
SCANS_FILE = ""      # the workbook made by 11_make_scans_xlsx.py
OUT_DIR = "validation_out"
# ----------------------------------------------------------------------------

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("ex8", HERE / "08_extract_pvc_sinus_from_scan.py")
ex8 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ex8)

LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
KEY_LEADS = ["V1", "V2", "V3"]
MATCH_WINDOW_MS = 200.0          # an annotated window is "matched" if an automatic
                                 # beat starts within this distance
BORDER_MS = 80.0                 # annotated windows this close to the panel border are
                                 # not evaluable (the automatic detector drops them)


# ------------------------------------------------------------------ helpers
def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def fmt_ci(k, n):
    lo, hi = wilson(k, n)
    return f"{k}/{n} = {k / max(n, 1):.3f}  (95% CI {lo:.3f}-{hi:.3f})"


def find_scans_file(given):
    """Explicit path wins. Otherwise: an .xlsx with a 'scans' sheet next to this
    script / in the current folder (scans.xlsx first, else the newest), then scans.csv."""
    if given:
        return Path(given)
    folders = []
    for f in (HERE, Path.cwd()):
        if f not in folders:
            folders.append(f)
    cands = []
    for f in folders:
        for p in f.glob("*.xlsx"):
            if p.name.startswith("~$"):              # Excel lock/temp files
                continue
            try:
                from openpyxl import load_workbook
                wb = load_workbook(p, read_only=True)
                has = "scans" in wb.sheetnames
                wb.close()
            except Exception:
                has = False
            if has:
                cands.append(p)
    if cands:
        cands.sort(key=lambda p: (p.name.lower() != "scans.xlsx", -p.stat().st_mtime))
        if len(cands) > 1:
            print("Several workbooks with a 'scans' sheet found; using the first:")
            for p in cands:
                print("   ", p)
        return cands[0]
    for f in folders:
        if (f / "scans.csv").exists():
            return f / "scans.csv"
    return None

def load_scans(path):
    """Read scans.xlsx (sheet 'scans') or scans.csv.
    Required: file, patient_id, gain_mV_per_cm.  Optional: scan_id, label,
    file_suffix / recording (informational only), gain_verified."""
    if str(path).lower().endswith((".xlsx", ".xlsm")):
        df = pd.read_excel(path, sheet_name="scans")
    else:
        df = pd.read_csv(path)
    df = df.dropna(subset=["file"]).copy()
    need = {"file", "patient_id", "gain_mV_per_cm"}
    miss = need - set(df.columns)
    if miss:
        sys.exit(f"scans file is missing columns: {sorted(miss)}")
    if "label" not in df.columns:
        df["label"] = np.nan
    if "scan_id" not in df.columns or df["scan_id"].isna().any():
        df["scan_id"] = df["file"].map(lambda f: Path(str(f)).stem)
    df["scan_id"] = df["scan_id"].astype(str)
    sfx = "file_suffix" if "file_suffix" in df.columns else ("recording" if "recording" in df.columns else None)
    df["recording"] = df[sfx] if sfx else np.nan          # informational only, never used for identity
    df["patient_id"] = df["patient_id"].astype(str)
    warns = []
    if df["scan_id"].duplicated().any():
        warns.append(f"duplicate scan_id(s): {df[df.scan_id.duplicated()].scan_id.tolist()}")
    for i, r in df.iterrows():
        if not Path(str(r.file)).exists():
            warns.append(f"file not found: {r.file}")
        if pd.isna(r.gain_mV_per_cm) or float(r.gain_mV_per_cm) <= 0:
            warns.append(f"{r.scan_id}: gain missing/invalid")
        if isinstance(r.label, str) and r.label not in ("LVOT", "RVOT"):
            warns.append(f"{r.scan_id}: label '{r.label}' is not LVOT/RVOT")
    if "gain_verified" in df.columns:
        gv = df["gain_verified"].fillna("").astype(str).str.strip().str.lower()
        nv = int((gv != "yes").sum())
        if nv:
            warns.append(f"{nv} of {len(df)} scans have gain_verified != 'yes'. A wrong gain scales "
                         f"every amplitude of that scan; check the header crops in the Excel file.")
    lab = df.dropna(subset=["label"]).groupby("patient_id").label.nunique()
    for pid in lab[lab > 1].index:
        warns.append(f"patient {pid} has conflicting labels")
    return df, warns


def run_scan(row, out_dir, force):
    d = out_dir / "scans" / row.scan_id
    info_path = d / "info.json"
    if info_path.exists() and not force:
        return json.loads(info_path.read_text()), "cached"
    try:
        label = row.label if isinstance(row.label, str) else None
        rec = row.recording if pd.notna(row.recording) else None
        ex8.process_scan(row.file, str(row.patient_id), d, label,
                         float(row.gain_mV_per_cm), rec, scan_id=row.scan_id)
        info = ex8.process_scan.last_info
        info_path.write_text(json.dumps(info))
        return info, "ok"
    except Exception as e:                                   # keep going on a bad scan
        d.mkdir(parents=True, exist_ok=True)
        (d / "ERROR.txt").write_text(traceback.format_exc())
        return {"error": f"{type(e).__name__}: {e}"}, "error"


def collapse_windows(ann):
    """The 12 panels are simultaneous, so one annotated window shows up once per column
    (up to 4 identical rows). The statistical unit is the UNIQUE window: kind is correct if
    MORE than half of its matched columns are correct, onset error = median over columns."""
    a = ann[(~ann.border) & ann.matched].copy()
    a["wkey"] = a.scan_id.astype(str) + "|" + a.ann_start_ms.round(0).astype(int).astype(str)
    g = a.groupby("wkey")
    w = g.agg(scan_id=("scan_id", "first"), ann_kind=("ann_kind", "first"),
              ann_start_ms=("ann_start_ms", "first"), n_cols=("kind_ok", "size"),
              frac_ok=("kind_ok", "mean"), onset_err_ms=("onset_err_ms", "median")).reset_index()
    w["kind_ok"] = w.frac_ok > 0.5
    return w


def cluster_boot_mean(values, groups, n=2000, seed=0):
    """95% CI of a mean when rows are clustered (scans/patients), by resampling clusters."""
    rng = np.random.default_rng(seed)
    v = pd.Series(np.asarray(values, float))
    gr = pd.Series(np.asarray(groups))
    idx = {k: np.where(gr.values == k)[0] for k in gr.unique()}
    keys = list(idx)
    out = []
    for _ in range(n):
        pick = rng.choice(len(keys), len(keys), replace=True)
        ii = np.concatenate([idx[keys[j]] for j in pick])
        out.append(v.values[ii].mean())
    return tuple(np.percentile(out, [2.5, 97.5]))


# ------------------------------------------------------- A: annotation agreement
def annotation_rows(scan_id, info):
    rows = []
    ppm = info["px_per_ms"]
    for col, ann_auto in info.get("auto_for_annotated", {}).items():
        ann = info["columns"][col]["beats"]                 # cardiologist windows (used as truth)
        for w in ann:
            s_ms, e_ms = w["start"] / ppm, w["end"] / ppm
            border = s_ms < BORDER_MS or e_ms > info["panel_ms"] - BORDER_MS
            cand = [(abs(a["start"] - w["start"]) / ppm, a) for a in ann_auto]
            d, a = min(cand, key=lambda t: t[0]) if cand else (1e9, None)
            matched = a is not None and d <= MATCH_WINDOW_MS
            rows.append({"scan_id": scan_id, "column": int(col), "ann_kind": w["kind"],
                         "ann_start_ms": s_ms, "ann_end_ms": e_ms, "border": border,
                         "matched": matched,
                         "auto_kind": a["kind"] if matched else "",
                         "kind_ok": bool(matched and a["kind"] == w["kind"]),
                         "onset_err_ms": (a["start"] - w["start"]) / ppm if matched else np.nan})
    return rows


# --------------------------------------------------------------- C: plausibility
def plausibility_rows(scan_id, info):
    """sinus, sinus, PVC triplets (by time) -> is the PVC premature?
       also: PVC QRS wider than sinus beats of the same column?"""
    rows = []
    ppm = info["px_per_ms"]
    for col, c in info["columns"].items():
        beats = info["auto_for_annotated"].get(col) if c["source"] == "cardiologist_lines" \
            else c["beats"]
        beats = sorted([b for b in (beats or []) if "peak" in b], key=lambda b: b["peak"])
        for i in range(2, len(beats)):
            a, b, p = beats[i - 2], beats[i - 1], beats[i]
            if a["kind"] == "sinus" and b["kind"] == "sinus" and p["kind"] == "PVC":
                # onset-to-onset: the activity peak of a wide PVC lies later than its onset
                rr = (b["start"] - a["start"]) / ppm
                cp = (p["start"] - b["start"]) / ppm
                rows.append({"scan_id": scan_id, "column": int(col), "test": "premature",
                             "value": cp / rr, "ok": cp < 0.9 * rr})
        sw = [b["width_ms"] for b in beats if b["kind"] == "sinus" and "width_ms" in b]
        pw = [b["width_ms"] for b in beats if b["kind"] == "PVC" and "width_ms" in b]
        if sw and pw:
            rows.append({"scan_id": scan_id, "column": int(col), "test": "wider",
                         "value": np.median(pw) / np.median(sw),
                         "ok": np.median(pw) > 1.15 * np.median(sw)})
    return rows


# ------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scans", default=SCANS_FILE or None)
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--force", action="store_true", help="recompute even if cached")
    ap.add_argument("--limit", type=int, default=None, help="process only the first N scans")
    ap.add_argument("--calib-frac", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--audit-n", type=int, default=15)
    ap.add_argument("--audit-summary", default=None,
                    help="filled audit_sheet.csv -> accuracy with Wilson CI, then exit")
    a = ap.parse_args()

    if a.audit_summary:
        return audit_summary(a.audit_summary)
    scans_path = find_scans_file(a.scans)
    if scans_path is None or not scans_path.exists():
        sys.exit("scans workbook not found. Put scans.xlsx (sheet 'scans') next to this script, "
                 "or pass it:  python 10_validate_extraction.py --scans \"E:\\path\\scans.xlsx\"")
    print(f"Reading scans from: {scans_path.resolve()}")

  
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    scans, warns = load_scans(scans_path)
    if a.limit:
        scans = scans.head(a.limit)
    pid_of = dict(zip(scans.scan_id, scans.patient_id))
    print(f"{len(scans)} scans, {scans.patient_id.nunique()} patients")
    for w in warns:
        print("WARNING:", w)

    # ---------------- run extraction on every scan
    infos, status = {}, {}
    t0 = time.time()
    for i, r in enumerate(scans.itertuples(index=False), start=1):
        info, st = run_scan(r, out, a.force)
        infos[r.scan_id], status[r.scan_id] = info, st
        print(f"[{i}/{len(scans)}] {r.scan_id}: {st}"
              + (f"  {info.get('error', '')}" if st == "error" else "")
              + f"   ({time.time() - t0:.0f}s)")
    ok_ids = [s for s in infos if "error" not in infos[s]]

    # ---------------- per-scan summary + flags
    ann_rows, pl_rows = [], []
    for sid in ok_ids:
        ann_rows += annotation_rows(sid, infos[sid])
        pl_rows += plausibility_rows(sid, infos[sid])
    ann = pd.DataFrame(ann_rows)
    pl = pd.DataFrame(pl_rows)

    recs = []
    for r in scans.itertuples(index=False):
        sid = r.scan_id
        inf = infos[sid]
        if "error" in inf:
            recs.append({"scan_id": sid, "patient_id": r.patient_id,
                         "label": r.label if isinstance(r.label, str) else "",
                         "status": "ERROR", "flags": "processing_error: " + inf["error"]})
            continue
        unrel = [l for l in LEADS if inf["unreliable"].get(l, True)]
        cols_with_pvc = sum(1 for c in inf["columns"].values()
                            if any(b["kind"] == "PVC" for b in c["beats"]))
        a_sc = ann[ann.scan_id == sid] if len(ann) else ann
        a_eval = a_sc[(a_sc.matched)] if len(a_sc) else a_sc
        p_sc = pl[pl.scan_id == sid] if len(pl) else pl
        flags = []
        if inf["n_events"] == 0:
            flags.append("no_PVC_event_extracted")
        if 12 - len(unrel) < 9:
            flags.append(f"only_{12 - len(unrel)}_reliable_leads")
        miss_key = [l for l in KEY_LEADS if l in unrel]
        if miss_key:
            flags.append("key_lead_unreliable:" + "+".join(miss_key))
        if abs(inf["theta"]) > 3:
            flags.append(f"large_skew_{inf['theta']:+.1f}deg")
        if cols_with_pvc < 3:
            flags.append(f"PVC_found_in_{cols_with_pvc}_of_4_columns")
        if inf["n_crops"] and inf["n_clipped"] / inf["n_crops"] > 0.25:
            flags.append("many_clipped_crops")
        if len(a_eval) and (~a_eval.kind_ok).any():
            flags.append("kind_mismatch_vs_cardiologist")
        if len(p_sc) and (~p_sc.ok).any():
            flags.append("implausible_PVC_timing_or_width")
        srcs = {c["source"] for c in inf["columns"].values()}
        win_src = "cardiologist_lines" if srcs == {"cardiologist_lines"} else "/".join(sorted(srcs))
        if win_src != "cardiologist_lines":
            flags.append("windows_NOT_from_cardiologist_lines(" + win_src + ")")
        recs.append({"scan_id": sid, "patient_id": r.patient_id,
                     "label": r.label if isinstance(r.label, str) else "",
                     "window_source": win_src,
                     "status": status[sid], "gain": inf["gain"], "deskew_deg": inf["theta"],
                     "reliable_leads": 12 - len(unrel), "unreliable_leads": " ".join(unrel),
                     "pvc_events": inf["n_events"], "crops": inf["n_crops"],
                     "clipped_frac": inf["n_clipped"] / max(inf["n_crops"], 1),
                     "cols_with_PVC": cols_with_pvc,
                     "annotated_windows_matched": int(len(a_eval)),
                     "kind_correct": int(a_eval.kind_ok.sum()) if len(a_eval) else 0,
                     "flags": "; ".join(flags)})
    summ = pd.DataFrame(recs)
    summ.to_csv(out / "per_scan_summary.csv", index=False)
    if len(ann):
        ann.to_csv(out / "annotation_agreement.csv", index=False)
    if len(pl):
        pl.to_csv(out / "plausibility.csv", index=False)

    # ---------------- lead reliability
    lr = []
    for l in LEADS:
        n = len(ok_ids)
        bad = sum(1 for s in ok_ids if infos[s]["unreliable"].get(l, True))
        lr.append({"lead": l, "scans": n, "unreliable": bad, "pct_unreliable": 100 * bad / max(n, 1)})
    lr = pd.DataFrame(lr)
    lr.to_csv(out / "lead_reliability.csv", index=False)
    plt.figure(figsize=(7, 3.5))
    plt.bar(lr.lead, lr.pct_unreliable, color=["#c0392b" if l in KEY_LEADS else "#7f8c8d" for l in lr.lead])
    plt.ylabel("% scans with panel excluded")
    plt.title("Panel reliability per lead (red = V1-V3)")
    plt.tight_layout()
    plt.savefig(out / "lead_reliability.png", dpi=130)
    plt.close()

    # ---------------- A/B: agreement + held-out calibration
    lines = []
    lines.append("# Extraction validation report\n")
    sha = lambda f: hashlib.sha1((HERE / f).read_bytes()).hexdigest()[:10]
    lines.append(f"Code fingerprint (sha1, first 10): 08={sha('08_extract_pvc_sinus_from_scan.py')}  "
                 f"10={sha('10_validate_extraction.py')}  -> if two runs differ, the files differ; "
                 f"USE_TIMING_PRIOR={ex8.USE_TIMING_PRIOR}, SHARED_TIME_AXIS={ex8.SHARED_TIME_AXIS}, REF_W={ex8.REF_W}, "
                 f"PVC_ONSET_SHIFT_MS={ex8.PVC_ONSET_SHIFT_MS}, SINUS_ONSET_SHIFT_MS={ex8.SINUS_ONSET_SHIFT_MS}\n")
    lines.append(f"Scans processed: {len(ok_ids)} / {len(scans)} "
                 f"(errors: {sum(1 for s in infos if 'error' in infos[s])}), "
                 f"patients: {scans.patient_id.nunique()}\n")
    checks = []

    if len(ann):
        ev = ann[(~ann.border)]
        m = collapse_windows(ann)                              # unique windows
        lines.append("\n## A. Agreement with the cardiologist's lines\n")
        lines.append(f"Annotation rows: {len(ann)} (one per panel column; the 4 columns are simultaneous, so a "
                     f"window repeats). UNIQUE annotated windows: {ann.assign(k=ann.scan_id.astype(str) + ann.ann_start_ms.round(0).astype(str)).k.nunique()}; "
                     f"border-truncated rows (not evaluable): {int(ann.border.sum())}; "
                     f"evaluable unique windows matched by an automatic beat: {len(m)}\n")
        lo, hi = cluster_boot_mean(m.kind_ok.astype(float), m.scan_id)
        lines.append(f"PVC/sinus classification correct (unit = unique window): {fmt_ci(int(m.kind_ok.sum()), len(m))}; "
                     f"scan-cluster bootstrap 95% CI {lo:.3f}-{hi:.3f}\n")
        for k in ("PVC", "sinus"):
            mk = m[m.ann_kind == k]
            lines.append(f"  - annotated {k}: {fmt_ci(int(mk.kind_ok.sum()), len(mk))}\n")
        wrong = m[~m.kind_ok]
        if len(wrong):
            ws = wrong.groupby("scan_id").size().sort_values(ascending=False)
            lines.append(f"Wrong windows: {len(wrong)} in {len(ws)} of {m.scan_id.nunique()} scans; "
                         f"worst scans: {ws.head(5).to_dict()}  (open scans/<id>/qc_windows.png)\n")
        agree = (m.frac_ok.isin([0.0, 1.0])).mean()
        lines.append(f"Columns agree with each other for {100 * agree:.0f}% of windows "
                     f"(errors that repeat in all columns are systematic, not lead noise)\n")
        err = m.onset_err_ms.dropna()
        if len(err):
            lines.append(f"Onset error (auto - cardiologist), ms: median {err.median():+.0f}, "
                         f"IQR [{err.quantile(.25):+.0f}, {err.quantile(.75):+.0f}], "
                         f"|err| median {err.abs().median():.0f}, P90 {err.abs().quantile(.9):.0f}, "
                         f"max {err.abs().max():.0f}\n")
            checks.append(("PVC/sinus kind accuracy >= 0.95 (lower CI >= 0.85)",
                           m.kind_ok.mean() >= 0.95 and lo >= 0.85, f"{m.kind_ok.mean():.3f} (CI {lo:.3f}-{hi:.3f})"))
            checks.append(("median |onset error| <= 30 ms",
                           err.abs().median() <= 30, f"{err.abs().median():.0f} ms"))
            checks.append(("P90 |onset error| <= 100 ms",
                           err.abs().quantile(.9) <= 100, f"{err.abs().quantile(.9):.0f} ms"))
        unm = ev[~ev.matched]
        if len(unm):
            lines.append(f"Unmatched annotation rows (no automatic beat nearby): {len(unm)} -> "
                         f"annotation_agreement.csv (matched == False)\n")
        m.to_csv(out / "annotation_agreement_unique_windows.csv", index=False)
        plt.figure(figsize=(6.5, 3.5))
        for k, c in (("PVC", "#c0392b"), ("sinus", "#27ae60")):
            e = m[m.ann_kind == k].onset_err_ms.dropna()
            if len(e):
                plt.hist(e, bins=np.arange(-210, 215, 20), alpha=0.6, label=f"{k} (n={len(e)})", color=c)
        plt.axvline(0, color="k", lw=0.8)
        plt.xlabel("onset error, ms (auto - cardiologist)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out / "onset_error_hist.png", dpi=130)
        plt.close()

        # held-out calibration (split by PATIENT)
        pats = sorted(m.scan_id.map(pid_of).unique())
        rng = np.random.default_rng(a.seed)
        n_cal = max(1, int(round(a.calib_frac * len(pats)))) if len(pats) >= 4 else 0
        lines.append("\n## B. Held-out calibration of window onsets\n")
        if n_cal == 0:
            lines.append("Too few annotated patients for a calibration/test split (need >= 4).\n")
        else:
            cal_p = set(rng.choice(pats, n_cal, replace=False))
            m = m.assign(patient=m.scan_id.map(pid_of))
            cal, tst = m[m.patient.isin(cal_p)], m[~m.patient.isin(cal_p)]
            shifts = {k: float(cal[cal.ann_kind == k].onset_err_ms.median())
                      for k in ("PVC", "sinus") if (cal.ann_kind == k).sum() >= 3}
            lines.append(f"Calibration patients: {len(cal_p)} ({len(cal)} windows); "
                         f"test patients: {len(pats) - len(cal_p)} ({len(tst)} windows); seed {a.seed}\n")
            for k in ("PVC", "sinus"):
                t = tst[tst.ann_kind == k].onset_err_ms.dropna()
                if len(t) == 0 or k not in shifts:
                    continue
                lines.append(f"  - {k}: median error on calibration set {shifts[k]:+.0f} ms. "
                             f"TEST set |error| median before {t.abs().median():.0f} ms -> "
                             f"after correcting by the calibration median {(t - shifts[k]).abs().median():.0f} ms\n")
            sug = []
            for k, const, cur in (("PVC", "PVC_ONSET_SHIFT_MS", ex8.PVC_ONSET_SHIFT_MS),
                                  ("sinus", "SINUS_ONSET_SHIFT_MS", ex8.SINUS_ONSET_SHIFT_MS)):
                t = tst[tst.ann_kind == k].onset_err_ms.dropna()
                if k in shifts and len(t) >= 5 and (t - shifts[k]).abs().median() < 0.8 * t.abs().median():
                    sug.append(f"{const} = {cur - shifts[k]:.0f}   (currently {cur:.0f})")
            if sug:
                lines.append("\nCalibration clearly improved the TEST-set error. Suggested constants for "
                             "08_extract_pvc_sinus_from_scan.py (edit, then rerun with --force):\n")
                lines.append("```\n" + "\n".join(sug) + "\n```\n")
            else:
                lines.append("\nNo constant is suggested: correcting by the calibration median did not clearly "
                             "improve the test-set error (or too few test windows). Keep the current constants.\n")
    else:
        lines.append("\n## A/B. No cardiologist lines found in any scan; skipped.\n")

    # ---------------- C: plausibility
    lines.append("\n## C. Physiological plausibility (all columns, with or without lines)\n")
    if len(pl):
        for t, nm in (("premature", "PVC premature (coupling < 0.9 x preceding sinus RR)"),
                      ("wider", "PVC QRS excursion > 1.15 x sinus beats")):
            x = pl[pl.test == t]
            if len(x):
                lines.append(f"  - {nm}: {fmt_ci(int(x.ok.sum()), len(x))}\n")
                checks.append((f"plausibility '{t}' >= 0.85", x.ok.mean() >= 0.85, f"{x.ok.mean():.3f}"))
    else:
        lines.append("No sinus-sinus-PVC triplets available.\n")

    # ---------------- D: quality
    s_ok = summ[summ.status != "ERROR"]
    lines.append("\n## D. Quality\n")
    lines.append(f"Unique PVC events per scan: median {s_ok.pvc_events.median():.0f}, range "
                 f"{int(s_ok.pvc_events.min())}-{int(s_ok.pvc_events.max())} (all 12 leads of a PVC share one event_id)\n")
    n_lines = int((s_ok.window_source == "cardiologist_lines").sum())
    lines.append(f"Scans whose windows come from the cardiologist's lines: {n_lines}/{len(s_ok)} "
                 f"(only these are 'ground truth' windows; the others used the automatic detector)\n")
    lines.append(f"Scans with >=1 PVC event extracted: {fmt_ci(int((s_ok.pvc_events > 0).sum()), len(s_ok))}\n")
    lines.append(f"Scans with >=9 reliable leads: {fmt_ci(int((s_ok.reliable_leads >= 9).sum()), len(s_ok))}\n")
    for l in KEY_LEADS:
        row = lr[lr.lead == l].iloc[0]
        lines.append(f"Lead {l} excluded in {int(row.unreliable)}/{int(row.scans)} scans\n")
    lines.append(f"Deskew angle: median {s_ok.deskew_deg.median():+.2f} deg, max |{s_ok.deskew_deg.abs().max():.2f}| deg\n")
    lines.append(f"Clipped crops: {s_ok.clipped_frac.mul(s_ok.crops).sum():.0f} of {int(s_ok.crops.sum())} "
                 f"({100 * s_ok.clipped_frac.mul(s_ok.crops).sum() / max(s_ok.crops.sum(), 1):.1f}%)\n")
    checks.append(("scans with >=1 PVC event >= 0.95", (s_ok.pvc_events > 0).mean() >= 0.95,
                   f"{(s_ok.pvc_events > 0).mean():.3f}"))
    checks.append(("scans with >=9 reliable leads >= 0.90", (s_ok.reliable_leads >= 9).mean() >= 0.90,
                   f"{(s_ok.reliable_leads >= 9).mean():.3f}"))
    checks.append(("each of V1,V2,V3 reliable in >= 85% scans",
                   all((100 - lr[lr.lead == l].pct_unreliable.iloc[0]) >= 85 for l in KEY_LEADS),
                   " / ".join(f"{l}:{100 - lr[lr.lead == l].pct_unreliable.iloc[0]:.0f}%" for l in KEY_LEADS)))

    # ---------------- F: are the windows aligned with the QRS in EVERY lead?
    lines.append("\n## F. Window alignment per lead (does the crop contain the QRS?)\n")
    parts = []
    for sid in ok_ids:
        f = out / "scans" / sid / "beats_extracted.csv"
        if f.exists():
            parts.append(pd.read_csv(f))
    allb = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    if len(allb) and "qrs_peak_offset_ms" in allb.columns:
        allb = allb.dropna(subset=["qrs_peak_offset_ms"])
        g = (allb.groupby(["beat", "lead"]).qrs_peak_offset_ms
             .agg(n="size", median="median",
                  pct_before_start=lambda x: 100 * (x < 0).mean(),
                  pct_lt_20ms=lambda x: 100 * (x < 20).mean()).reset_index())
        g.to_csv(out / "window_alignment.csv", index=False)
        lines.append("Offset = time of the QRS peak relative to the START of the annotated window "
                     "(negative = the QRS lies before the window). Crops start 40 ms before the window.\n")
        lines.append("```\n" + g.round(1).to_string(index=False) + "\n```\n")
        bad = g[g.pct_before_start > 10]
        if len(bad):
            lines.append("Leads where > 10% of crops have the QRS BEFORE the window start: "
                         + ", ".join(f"{r.lead}/{r.beat} ({r.pct_before_start:.0f}%)" for r in bad.itertuples()) + "\n")
        worst = (allb[allb.qrs_peak_offset_ms < -20].groupby("scan_id").size()
                 .sort_values(ascending=False).head(8))
        if len(worst):
            lines.append(f"Scans with most crops whose QRS is >20 ms before the window: {worst.to_dict()} "
                         f"(open scans/<id>/qc_overlay.jpg)\n")
        checks.append(("QRS peak inside window (offset >= 0) in >= 90% of crops, every lead/beat",
                       bool((g.pct_before_start <= 10).all()),
                       f"worst {g.pct_before_start.max():.0f}%"))
    else:
        lines.append("No alignment data (rerun with --force so the per-crop offsets are written).\n")

    # ---------------- checks table
    lines.append("\n## Automatic acceptance checks (thresholds are suggestions, edit as you wish)\n")
    lines.append("```\n")
    for nm, ok, v in checks:
        lines.append(f"{'PASS' if ok else 'FAIL'}  {nm:55s} observed {v}\n")
    lines.append("```\n")

    # ---------------- review list
    rv = summ[summ["flags"].fillna("") != ""].copy()
    rv["n_flags"] = rv["flags"].str.count(";") + 1
    rv = rv.sort_values("n_flags", ascending=False)
    rv.to_csv(out / "review_list.csv", index=False)
    lines.append(f"\n## E. Scans to review manually: {len(rv)} of {len(summ)} "
                 f"(review_list.csv; open scans/<scan_id>/qc_windows.png)\n")

    # ---------------- audit sheet (random scans, stratified by label)
    rng = np.random.default_rng(a.seed + 1)
    pool = summ[summ.status != "ERROR"]
    take = []
    if pool.label.replace("", np.nan).notna().any():
        for lab, g in pool.groupby(pool.label.replace("", "NA")):
            k = max(1, int(round(a.audit_n * len(g) / len(pool))))
            take += list(rng.choice(g.scan_id.values, min(k, len(g)), replace=False))
    else:
        take = list(rng.choice(pool.scan_id.values, min(a.audit_n, len(pool)), replace=False))
    arows = []
    for sid in take:
        inf = infos[sid]
        cols_ = inf["columns"]
        sig = {tuple((b["kind"], round(b["start"]), round(b["end"])) for b in v["beats"]) for v in cols_.values()}
        shared = len(sig) == 1                                   # simultaneous leads, identical windows
        items = [("all", next(iter(cols_.values())))] if shared else list(cols_.items())
        for c, v in items:
            arows.append({"scan_id": sid, "column": c, "detection_source": v["source"],
                          "n_PVC_windows": sum(b["kind"] == "PVC" for b in v["beats"]),
                          "n_sinus_windows": sum(b["kind"] == "sinus" for b in v["beats"]),
                          "qc_image": f"scans/{sid}/qc_overlay.jpg",
                          "pvc_correct(1/0)": "", "sinus_correct(1/0)": "", "notes": ""})
    pd.DataFrame(arows).sort_values(["scan_id", "column"]).to_csv(out / "audit_sheet.csv", index=False)
    lines.append(f"Audit sheet: {len(take)} random scans (one row per scan when all 12 leads share the same windows) -> audit_sheet.csv "
                 f"(fill the two 1/0 columns, then run --audit-summary)\n")

    (out / "validation_report.md").write_text("".join(lines), encoding="utf-8")
    print("\n" + "".join(lines))

    # ---------------- PHI-free share bundle
    sb = out / "share_bundle"
    sb.mkdir(exist_ok=True)
    for f in ("validation_report.md", "lead_reliability.csv", "lead_reliability.png",
              "onset_error_hist.png", "annotation_agreement.csv", "plausibility.csv",
              "review_list.csv"):
        if (out / f).exists():
            shutil.copy(out / f, sb / f)
    shutil.copy(out / "per_scan_summary.csv", sb / "per_scan_summary.csv")
    for sid in rv.scan_id.head(12):                                   # QC plots of the worst scans
        q = out / "scans" / sid / "qc_windows.png"
        if q.exists():
            shutil.copy(q, sb / f"qc_{sid}.png")
    print(f"\nShare bundle (check it contains no names / MRN before sending): {sb}")


def audit_summary(path):
    d = pd.read_csv(path)
    for col in ("pvc_correct(1/0)", "sinus_correct(1/0)"):
        d[col] = pd.to_numeric(d[col], errors="coerce")
    n_missing = int(d["pvc_correct(1/0)"].isna().sum())
    if n_missing:
        print(f"NOTE: {n_missing} rows are not filled and are ignored")
    print("\nManual audit of the automatic windows (unit = scan column)")
    for col, nm in (("pvc_correct(1/0)", "PVC windows correct"),
                    ("sinus_correct(1/0)", "sinus windows correct")):
        x = d[col].dropna()
        print(f"  {nm}: {fmt_ci(int(x.sum()), len(x))}")
    both = d.dropna(subset=["pvc_correct(1/0)", "sinus_correct(1/0)"])
    ok = int(((both["pvc_correct(1/0)"] == 1) & (both["sinus_correct(1/0)"] == 1)).sum())
    print(f"  both correct: {fmt_ci(ok, len(both))}")
    # cluster-aware note: columns of one scan are not independent
    print(f"  (scans audited: {d.dropna(subset=['pvc_correct(1/0)']).scan_id.nunique()})")
    wrong = d[(d["pvc_correct(1/0)"] == 0) | (d["sinus_correct(1/0)"] == 0)]
    if len(wrong):
        print("\nColumns marked wrong (look at their qc images):")
        print(wrong[["scan_id", "column", "detection_source", "notes"]].to_string(index=False))


if __name__ == "__main__":
    main()
