"""
09_batch_extract.py
-------------------
Run 08_extract_pvc_sinus_from_scan.py on every scan listed in a CSV and build
ONE manifest (patient_id, label, lead, beat type, file, mV amplitudes ...).

scans.xlsx / scans.csv  (see 11_make_scans_xlsx.py), columns:
  file, patient_id, gain_mV_per_cm  (+ scan_id, label, file_suffix, gain_verified)
    file            path to the scan image
    patient_id      the subject key used for ALL later splitting (e.g. P11)
    recording       recording index of that patient (3 for "11_3")
    label           LVOT / RVOT (leave empty if unknown)
    gain_mV_per_cm  value printed in the header of the printout (1.0 or 2.0)

Usage:  python 09_batch_extract.py scans.xlsx OUT_DIR
"""
import importlib.util
import sys
from pathlib import Path

import pandas as pd

here = Path(__file__).parent
spec = importlib.util.spec_from_file_location("ex8", here / "08_extract_pvc_sinus_from_scan.py")
ex8 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ex8)

spec2 = importlib.util.spec_from_file_location("val10", here / "10_validate_extraction.py")
val10 = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(val10)
scans, warns = val10.load_scans(sys.argv[1])
for w in warns:
    print("WARNING:", w)
out = Path(sys.argv[2])
all_rows, summary = [], []
for _, r in scans.iterrows():
    sub = out / str(r.scan_id)
    label = r.label if isinstance(r.label, str) and r.label else None
    rec = r.recording if pd.notna(r.recording) else None
    df, qc, _ = ex8.process_scan(r.file, str(r.patient_id), sub, label,
                                 float(r.gain_mV_per_cm), rec, scan_id=r.scan_id)
    all_rows.append(df)
    n_pvc = int(df.event_id.nunique()) if len(df) else 0          # unique PVC events (12 leads are simultaneous)
    summary.append({"scan_id": r.scan_id, "patient_id": r.patient_id, "label": label,
                    "unique_pvc_events": n_pvc, "crops": len(df),
                    "leads_with_crops": df.lead.nunique() if len(df) else 0})
manifest = pd.concat(all_rows, ignore_index=True)
manifest.to_csv(out / "beats_manifest_all.csv", index=False)
print(pd.DataFrame(summary).to_string(index=False))
print("\ntotal crops:", len(manifest), "| scans:", manifest.scan_id.nunique(),
      "| patients:", manifest.patient_id.nunique())
