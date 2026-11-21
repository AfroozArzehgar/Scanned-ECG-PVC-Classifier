#!/usr/bin/env python3
"""
15_lead_ablation.py
-------------------
Lead-ablation experiment for the deep patient-level models of 14_train_deep_patient_level.py.

Every model is RE-TRAINED from scratch in every condition (not just evaluated with a blanked input):

    input of the model      condition
    V1 + V2 + V3            baseline
    V2 + V3                 V1 removed
    V1 + V3                 V2 removed
    V1 + V2                 V3 removed

HOW A LEAD IS REMOVED  The removed lead is replaced by a blank channel in BOTH training and test data (images: 0,
    wave1d: the 2 envelope channels = 0). The network architectures (3 image channels / 6 wave channels, MobileNetV2
    needs 3) stay identical, and a constant-zero channel carries no information, so this equals dropping the lead
    while keeping the conditions comparable. (It is the same convention the pipeline already uses for a missing lead.)
    For wave1d with --wave-norm shape the lead is removed BEFORE the per-event normalisation; otherwise the scale
    factor (largest excursion of the event) would still carry information about the removed lead.

IDENTICAL SPLITS  All conditions use exactly the same patient folds and seeds as script 14, so the conditions are
    paired and the baseline row reproduces script 14 (up to GPU/TF non-determinism).

OUTPUT (--out)
    ablation_results.csv          one row per model x condition: AUC, BalAcc, Sens(RVOT), Spec(LVOT) with
                                  patient-bootstrap 95 % CI (+ Brier, PPV, NPV, confusion counts, paired AUC difference
                                  to the baseline with its patient-bootstrap CI)
    ablation_oof_predictions.csv  patient-level out-of-fold scores of every model|condition
    ablation_metrics.png/.svg     the 4 metrics with CI, per model and condition
    ablation_roc.png/.svg         ROC curves of the 4 conditions, one panel per model
    ablation_report.md

Usage
  python 15_lead_ablation.py --run-dir validation_out --out ablation_out \\
        --restrict-to features_out/patient_features.csv --ensemble 3
  python 15_lead_ablation.py --run-dir validation_out --out ablation_quick --quick --no-imagenet --models wave1d,custom
"""
import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve

# script 14 starts with a digit, so it cannot be imported with a normal import statement
_spec = importlib.util.spec_from_file_location("deep14", Path(__file__).resolve().parent / "14_train_deep_patient_level.py")
d = importlib.util.module_from_spec(_spec)
sys.modules["deep14"] = d
_spec.loader.exec_module(d)

# (label, short label, indices of the leads that are kept: 0 = V1, 1 = V2, 2 = V3)
CONDITIONS = [("V1+V2+V3 (baseline)", "baseline", (0, 1, 2)),
              ("V2+V3 (V1 removed)", "-V1", (1, 2)),
              ("V1+V3 (V2 removed)", "-V2", (0, 2)),
              ("V1+V2 (V3 removed)", "-V3", (0, 1))]


def mask_images(X, keep):
    Xc = X.copy()
    for c in range(3):
        if c not in keep:
            Xc[..., c] = 0.0
    return Xc


def mask_wave(W, keep):
    Wc = W.copy()
    for c in range(3):
        if c not in keep:
            Wc[..., 2 * c:2 * c + 2] = 0.0
    return Wc


def condition_data(model, X, Wraw, keep, wave_norm):
    if model == "wave1d":
        return d.normalize_wave(mask_wave(Wraw, keep), wave_norm)      # remove first, normalise afterwards
    return mask_images(X, keep)


def plot_metrics(res, models, out):
    items = [("AUC", "AUC", "CI_lo", "CI_hi"), ("BalAcc", "Balanced accuracy", "BalAcc_lo", "BalAcc_hi"),
             ("Sens(RVOT)", "Sensitivity (RVOT)", "Sens_lo", "Sens_hi"),
             ("Spec(LVOT)", "Specificity (LVOT)", "Spec_lo", "Spec_hi")]
    shorts = [c[1] for c in CONDITIONS]
    fig, axes = plt.subplots(2, 2, figsize=(9, 6.5), sharex=True)
    off = (np.arange(len(models)) - (len(models) - 1) / 2) * 0.22
    for ax, (col, title, lo, hi) in zip(axes.ravel(), items):
        for k, m in enumerate(models):
            s = res[res.model == m].set_index("short").loc[shorts]
            ax.errorbar(np.arange(len(shorts)) + off[k], s[col], yerr=[s[col] - s[lo], s[hi] - s[col]],
                        fmt="o", capsize=3, label=m)
        if col in ("AUC", "BalAcc"):
            ax.axhline(0.5, color="gray", ls=":", lw=0.8)
        ax.set_ylim(0, 1.02)
        ax.set_title(title, fontsize=10)
        ax.set_xticks(np.arange(len(shorts)))
        ax.set_xticklabels(shorts)
    axes[0, 0].legend(fontsize=8, loc="lower left")
    fig.suptitle("Lead ablation - re-trained models, patient-level OOF, patient-bootstrap 95 % CI", fontsize=10)
    plt.tight_layout()
    d.save_fig(fig, out, "ablation_metrics")


def plot_roc(oofs, yp, models, out):
    fig, axes = plt.subplots(1, len(models), figsize=(4.6 * len(models), 4.4), squeeze=False)
    for ax, m in zip(axes[0], models):
        for label, short, _ in CONDITIONS:
            p = oofs[(m, short)]
            f, t, _ = roc_curve(yp, p)
            ax.plot(f, t, label=f"{short} ({roc_auc_score(yp, p):.2f})")
        ax.plot([0, 1], [0, 1], ":", color="gray")
        ax.set_xlabel("1 - specificity")
        ax.set_ylabel("sensitivity (RVOT)")
        ax.set_title(m)
        ax.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    d.save_fig(fig, out, "ablation_roc")


def main():
    ap = argparse.ArgumentParser()
    d.add_common_args(ap)
    ap.add_argument("--out", default="ablation_out")
    a = ap.parse_args()
    a.cam_layer_mobilenet = "block_5_add"                                # unused here (no XAI), needed by cross_validate
    imagenet = d.finalize_args(a)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    X, Wraw, meta = d.load_events(a.run_dir, a.min_leads, a.restrict_to)   # the SAME events/patients for every condition
    pats, yp, ev_pat, ev_y, a.folds = d.prepare_patients(meta, a.folds)
    print(f"events {len(meta)}, patients {len(pats)} (LVOT {int((yp == 0).sum())}, RVOT {int((yp == 1).sum())})")
    models = [m.strip() for m in a.models.split(",")]

    rows, oofs = [], {}
    for m in models:
        for label, short, keep in CONDITIONS:
            print(f"\n=== {m} | {label}", flush=True)
            D = condition_data(m, X, Wraw, keep, a.wave_norm)
            oo, _, _, _ = d.cross_validate(m, D, yp, ev_pat, ev_y, a, imagenet, with_xai=False, label=f"{m}|{short}")
            mo = np.nanmean(oo, 0)
            aucs = [roc_auc_score(yp, o) for o in oo]
            row = d.summarize(yp, mo, a.threshold, a.n_boot, a.seed, aucs)
            rows.append({"model": m, "condition": label, "short": short, "leads_kept": "+".join(d.CHANNEL_LEADS[i] for i in keep), **row})
            oofs[(m, short)] = mo
            print(f"   AUC {row['AUC']:.3f} [{row['CI_lo']:.3f}, {row['CI_hi']:.3f}]  BalAcc {row['BalAcc']:.3f} "
                  f"[{row['BalAcc_lo']:.3f}, {row['BalAcc_hi']:.3f}]  Sens {row['Sens(RVOT)']:.3f}  Spec {row['Spec(LVOT)']:.3f}", flush=True)

    res = pd.DataFrame(rows)
    # paired AUC difference to the baseline of the same model (same patients, patient bootstrap)
    res["dAUC_vs_baseline"] = np.nan
    res["dAUC_lo"] = np.nan
    res["dAUC_hi"] = np.nan
    for m in models:
        base = oofs[(m, "baseline")]
        for _, short, _ in CONDITIONS[1:]:
            _, dci = d.boot_ci(yp, oofs[(m, short)], n=a.n_boot, seed=a.seed, p2=base)
            i = res.index[(res.model == m) & (res.short == short)][0]
            res.loc[i, ["dAUC_vs_baseline", "dAUC_lo", "dAUC_hi"]] = [
                roc_auc_score(yp, oofs[(m, short)]) - roc_auc_score(yp, base), dci[0], dci[1]]
    res.to_csv(out / "ablation_results.csv", index=False)
    pd.DataFrame({"patient_id": pats, "y_RVOT": yp, **{f"{m}|{s}": p for (m, s), p in oofs.items()}}) \
        .to_csv(out / "ablation_oof_predictions.csv", index=False)

    plot_metrics(res, models, out)
    plot_roc(oofs, yp, models, out)

    cols = ["model", "condition", "AUC", "CI_lo", "CI_hi", "BalAcc", "BalAcc_lo", "BalAcc_hi", "Sens(RVOT)", "Sens_lo",
            "Sens_hi", "Spec(LVOT)", "Spec_lo", "Spec_hi"]
    rep = ["# Lead-ablation report\n",
           f"patients {len(pats)} (LVOT {int((yp == 0).sum())}, RVOT {int((yp == 1).sum())}), events {len(meta)}; "
           f"CV {a.repeats} x {a.folds}-fold over patients (identical splits in all conditions); ensemble {a.ensemble}; "
           f"wave-norm {a.wave_norm}; ImageNet weights {imagenet}; fixed threshold {a.threshold}; seed {a.seed}\n",
           "\n## Performance (patient-bootstrap 95 % CI)\n```\n" + res[cols].round(3).to_string(index=False) + "\n```\n",
           "\n## Paired AUC difference to the baseline (condition minus baseline, patient bootstrap)\n```\n"
           + res[res.short != "baseline"][["model", "condition", "dAUC_vs_baseline", "dAUC_lo", "dAUC_hi"]]
           .round(3).to_string(index=False) + "\n```\n",
           "\nRead with care: the CIs resample patients only (the OOF scores are fixed), they do not include the "
           "training randomness; with ~54 patients they are wide, so differences between conditions are usually not "
           "distinguishable from zero unless the paired CI excludes 0. A lead that is not needed (AUC unchanged) is "
           "not proof that it carries no information - the other leads can be correlated with it. "
           "Sensitivity / specificity are at the fixed threshold, not at an optimised one.\n"]
    (out / "ablation_report.md").write_text("".join(rep), encoding="utf-8")
    print(f"\nreport: {(out / 'ablation_report.md').resolve()}")


if __name__ == "__main__":
    main()
