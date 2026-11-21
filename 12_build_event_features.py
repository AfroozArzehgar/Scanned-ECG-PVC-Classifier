#!/usr/bin/env python3
"""
12_build_event_features.py
--------------------------
From the extraction output (beats_extracted.csv + waveforms.npz of every scan) build

  event_features.csv    one row per PVC EVENT (all simultaneous leads of one PVC)
  patient_features.csv  one row per PATIENT (median over the patient's usable events)
  feature_report.txt    counts, exclusions, missingness

Everything is measured from the upper/lower envelope of the traced waveform, in mV
(gain-calibrated). The QRS part of each beat is isolated first (region where the vertical
extent of the trace exceeds 25 % of its maximum, gaps < 30 ms bridged), so that R and S are not
contaminated by the slow ST-T part of a wide PVC.

Per lead and beat type (pvc / sin):  R (mV above baseline), S (mV below baseline), Dur (ms).
Derived (log ratios use a 0.05 mV floor against noise):
  lR  = log(R_pvc / R_sin),  lS = log(S_pvc / S_sin),  RSp = log(R_pvc / S_pvc),
  RSs = log(R_sin / S_sin),  lDur = log(Dur_pvc / Dur_sin),
  dRS = RSp - RSs = log[(R/S)_pvc / (R/S)_sin]  (the V2 transition ratio idea, for every lead)
Clinical indices (MY implementation from memory of the published definitions; check them
against the original papers before you report them):
  V2S_V3R         = S_pvc(V2) / R_pvc(V3)                      (Yoshida et al.)
  V2_trans_ratio  = [R/S pvc in V2] / [R/S sinus in V2]        (Betensky et al.)
  tz_pvc, tz_sin  = first precordial lead V1..V6 with R >= S    (7 = none)
  tz_index        = tz_pvc - tz_sin                             (transition-zone index)

Usage
  python 12_build_event_features.py --run-dir validation_out --out features_out
  (--run-dir = the folder that contains <scan_id>/beats_extracted.csv, either the
   10_validate_extraction.py output (scans/<id>/...) or the 09_batch_extract.py output)
Options
  --min-leads 9      events with fewer leads that have a measurable PVC are dropped
  --drop-clipped     also drop events in which more than 25 % of the crops are clipped
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
PREC = ["V1", "V2", "V3", "V4", "V5", "V6"]
MIN_ACT_MV = 0.05        # below this vertical extent the lead has no measurable QRS
QRS_FRAC = 0.25
BRIDGE_MS = 30.0
EPS_MV = 0.05            # noise floor used inside log ratios


def qrs_measures(top, bot, dt_ms):
    """top/bot: envelopes (mV, relative to baseline). Returns R, S (mV, >=0) and Dur (ms)."""
    top = np.nan_to_num(np.asarray(top, float))
    bot = np.nan_to_num(np.asarray(bot, float))
    ext = top - bot
    if ext.size < 5 or ext.max() < MIN_ACT_MV:
        return {"R": np.nan, "S": np.nan, "Dur": np.nan}
    mask = ext > QRS_FRAC * ext.max()
    gap = max(1, int(round(BRIDGE_MS / dt_ms)))
    mask = ndi.binary_closing(np.pad(mask, gap), structure=np.ones(gap * 2 + 1))[gap:-gap]
    lab, _ = ndi.label(mask)
    comp = lab == lab[int(np.argmax(ext))]
    idx = np.where(comp)[0]
    l, r = max(0, idx.min() - 1), min(len(ext) - 1, idx.max() + 1)
    return {"R": max(0.0, float(top[l:r + 1].max())),
            "S": max(0.0, float(-bot[l:r + 1].min())),
            "Dur": (r - l + 1) * dt_ms}


def transition_lead(R, S):
    """first precordial lead (1..6) with R >= S among the AVAILABLE leads; 7 = none; nan if <4 leads."""
    avail = [(i + 1, R[l], S[l]) for i, l in enumerate(PREC) if not (np.isnan(R[l]) or np.isnan(S[l]))]
    if len(avail) < 4:
        return np.nan
    for i, r, s in avail:
        if r >= s:
            return float(i)
    return 7.0


def event_rows(run_dir):
    rows = []
    for csv in sorted(Path(run_dir).rglob("beats_extracted.csv")):
        d = pd.read_csv(csv)
        if d.empty or "event_id" not in d.columns:
            continue
        npz_path = csv.parent / "waveforms.npz"
        if not npz_path.exists():
            print(f"  skip {csv.parent.name}: no waveforms.npz (re-run extraction with the current 08)")
            continue
        wf = np.load(npz_path)
        info_dt = None
        scan_id = str(d.scan_id.iloc[0])
        # dt from the csv/info if present, else derive from the panel scale (217.4 px/s)
        dt_ms = 1000.0 / 217.44
        for ev, g in d.groupby("event_id"):
            feat = {"scan_id": scan_id, "patient_id": d.patient_id.iloc[0],
                    "label": d.label.iloc[0] if "label" in d.columns else np.nan,
                    "event_id": int(ev), "gain": float(d.gain_mV_per_cm.iloc[0])}
            R, S, Dur, clipped = {}, {}, {}, {}
            for lead in LEADS:
                gl = g[g.lead == lead]
                for kind, tag in (("PVC", "pvc"), ("sinus", "sin")):
                    gk = gl[gl.beat == kind]
                    m = {"R": np.nan, "S": np.nan, "Dur": np.nan}
                    for r in gk.itertuples():
                        key = f"{lead}|{kind}|c{int(r.column)}|p{int(r.pvc_index)}"
                        if key in wf.files:
                            a = wf[key]
                            m = qrs_measures(a[0], a[1], dt_ms)
                            if tag == "pvc":
                                clipped[lead] = bool(r.clipped)
                            break
                    feat[f"{lead}_{tag}R"], feat[f"{lead}_{tag}S"], feat[f"{lead}_{tag}Dur"] = m["R"], m["S"], m["Dur"]
                    if tag == "pvc":
                        R[lead], S[lead], Dur[lead] = m["R"], m["S"], m["Dur"]
                    else:
                        feat[f"{lead}_sinRS_tmp"] = m
            # derived per-lead ratios
            Rs = {l: feat[f"{l}_sinR"] for l in LEADS}
            Ss = {l: feat[f"{l}_sinS"] for l in LEADS}
            Ds = {l: feat[f"{l}_sinDur"] for l in LEADS}
            for l in LEADS:
                feat.pop(f"{l}_sinRS_tmp", None)
                pr, ps, pd_ = R[l], S[l], Dur[l]
                sr, ss, sd = Rs[l], Ss[l], Ds[l]
                lg = lambda a, b: float(np.log((a + EPS_MV) / (b + EPS_MV))) if not (np.isnan(a) or np.isnan(b)) else np.nan
                feat[f"{l}_lR"] = lg(pr, sr)
                feat[f"{l}_lS"] = lg(ps, ss)
                feat[f"{l}_RSp"] = lg(pr, ps)
                feat[f"{l}_RSs"] = lg(sr, ss)
                feat[f"{l}_dRS"] = (feat[f"{l}_RSp"] - feat[f"{l}_RSs"]
                                    if not (np.isnan(feat[f"{l}_RSp"]) or np.isnan(feat[f"{l}_RSs"])) else np.nan)
                feat[f"{l}_lDur"] = (float(np.log(pd_ / sd)) if not (np.isnan(pd_) or np.isnan(sd) or sd <= 0 or pd_ <= 0)
                                     else np.nan)
            # clinical indices
            feat["V2S_V3R"] = (S["V2"] / (R["V3"] + EPS_MV)) if not (np.isnan(S["V2"]) or np.isnan(R["V3"])) else np.nan
            if not any(np.isnan(x) for x in (R["V2"], S["V2"], Rs["V2"], Ss["V2"])):
                feat["V2_trans_ratio"] = ((R["V2"] + EPS_MV) / (S["V2"] + EPS_MV)) / ((Rs["V2"] + EPS_MV) / (Ss["V2"] + EPS_MV))
            else:
                feat["V2_trans_ratio"] = np.nan
            feat["tz_pvc"] = transition_lead(R, S)
            feat["tz_sin"] = transition_lead(Rs, Ss)
            feat["tz_index"] = (feat["tz_pvc"] - feat["tz_sin"]) if not (np.isnan(feat["tz_pvc"]) or np.isnan(feat["tz_sin"])) else np.nan
            # quality
            feat["n_leads_pvc"] = int(sum(not np.isnan(R[l]) for l in LEADS))
            feat["n_leads_sin"] = int(sum(not np.isnan(Rs[l]) for l in LEADS))
            feat["clipped_frac"] = float(np.mean(list(clipped.values()))) if clipped else np.nan
            rows.append(feat)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out", default="features_out")
    ap.add_argument("--min-leads", type=int, default=9)
    ap.add_argument("--drop-clipped", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    ev = event_rows(a.run_dir)
    if ev.empty:
        raise SystemExit("no events found: run 09/10 first with the current 08 (it writes waveforms.npz)")
    ev["usable"] = ev.n_leads_pvc >= a.min_leads
    if a.drop_clipped:
        ev["usable"] &= ~(ev.clipped_frac > 0.25)
    ev.to_csv(out / "event_features.csv", index=False)

    num = [c for c in ev.columns if c not in ("scan_id", "patient_id", "label", "event_id", "usable")]
    u = ev[ev.usable]
    pat = (u.groupby("patient_id")[num].median(numeric_only=True))
    pat["n_events_used"] = u.groupby("patient_id").size()
    pat["n_scans"] = u.groupby("patient_id").scan_id.nunique()
    lab = ev.groupby("patient_id").label.agg(lambda s: s.dropna().iloc[0] if s.notna().any() else np.nan)
    pat.insert(0, "label", lab.reindex(pat.index))
    pat = pat.reset_index()
    pat.to_csv(out / "patient_features.csv", index=False)

    key = ["V2S_V3R", "V2_trans_ratio", "tz_index"] + [f"{l}_{k}" for l in ("V1", "V2", "V3") for k in ("RSp", "dRS", "lR", "lS", "lDur")]
    rep = []
    rep.append(f"events: {len(ev)} total, {int(ev.usable.sum())} usable (>= {a.min_leads} leads"
               f"{', not clipped' if a.drop_clipped else ''}); patients with >= 1 usable event: {len(pat)} "
               f"of {ev.patient_id.nunique()}")
    rep.append(f"events per patient (usable): {u.groupby('patient_id').size().value_counts().sort_index().to_dict()}")
    if pat.label.notna().any():
        rep.append(f"label counts (patients): {pat.label.value_counts().to_dict()}")
    rep.append("\nmissing values in patient_features (key features):")
    for k in key:
        rep.append(f"  {k:16s} {int(pat[k].isna().sum())}/{len(pat)} missing")
    dropped = ev[~ev.usable]
    if len(dropped):
        rep.append(f"\nunusable events: {dropped[['scan_id', 'event_id', 'n_leads_pvc', 'clipped_frac']].round(2).to_dict('records')[:20]}")
    (out / "feature_report.txt").write_text("\n".join(rep), encoding="utf-8")
    print("\n".join(rep))
    print(f"\nwritten to {out.resolve()}")


if __name__ == "__main__":
    main()
