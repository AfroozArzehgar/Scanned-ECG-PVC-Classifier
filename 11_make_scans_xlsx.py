#!/usr/bin/env python3
"""
11_make_scans_xlsx.py
---------------------
Give it a FOLDER of scans; it writes  scans.xlsx  (and scans.csv) ready for
10_validate_extraction.py / 09_batch_extract.py.

  python 11_make_scans_xlsx.py --scans-dir E:\\ECG\\scans --out scans.xlsx

What it does automatically
  * finds every image (jpg / jpeg / png / tif) in the folder AND its sub-folders
  * patient_id  = the FIRST number in the file name        (11_3.jpeg -> P11)
  * file_suffix = whatever follows it (the "3" in 11_3). It is stored ONLY for
                  information and is NEVER used to identify the patient.
  * label       = taken from the folder name when the image sits in a folder whose
                  name contains LVOT or RVOT (e.g. LVOT_Final\\11_3.jpeg), or from
                  an optional --labels file (patient_id,label), else left blank
  * gain        = set to --gain-default and flagged UNVERIFIED. A small crop of the
                  printout's header (right side only: "25mm/sec 1.0mV/cm", NO name)
                  is embedded next to every row so you can check/correct it in seconds.
  * checks      = unreadable files, duplicate scans, conflicting/missing labels,
                  odd image sizes, and the distribution of file suffixes.

Sheets in the workbook:  scans (fill/verify this one) | patients | checks | readme
"""
import argparse
import io
import re
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}

# ----------------------------------------------------------------------------
# EDIT HERE if you press "Run" in VS Code (no terminal arguments).
# Leave SCANS_DIR empty and the script will ASK you for the folder instead.
SCANS_DIR = r"E:\ECG\PVC_Code\scans"                 # e.g. r"E:\ECG\scans"
OUT_XLSX = "scans.xlsx"
LABELS_FILE = ""                # optional: csv/xlsx with columns patient_id,label
# ----------------------------------------------------------------------------


def read_img(path):
    """Returns (img_bgr or None, error_message)."""
    err = ""
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        if data.size == 0:
            return None, "file is empty (0 bytes)"
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is not None:
            return img, ""
        err = "cv2 could not decode it"
    except Exception as e:
        err = f"cv2: {e!r}"
    try:
        from PIL import Image
        with Image.open(path) as im:
            arr = np.array(im.convert("RGB"))
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR), ""
    except Exception as e:
        return None, f"{err}; PIL: {e!r}"


def parse_name(stem):
    """first integer = patient; whatever follows (e.g. '.3', '_3', '-3') = informational suffix."""
    m = re.match(r"^\s*[Pp]?(\d+)(?:[\s_\-\.\(]+(.*?)\)?)?\s*$", stem)
    if not m:
        return None, ""
    return m.group(1), (m.group(2) or "")


def label_from_path(path, root):
    parts = [p.upper() for p in path.relative_to(root).parts[:-1]]
    found = {k for p in parts for k in ("LVOT", "RVOT") if k in p}
    if len(found) == 1:
        return found.pop()
    if len(found) > 1:
        return "CONFLICT"
    return ""


def header_crop_png(img):
    """Right-hand part of the top header only (gain text). Left part (name, MRN,
    date) is deliberately NOT included."""
    H, W = img.shape[:2]
    crop = img[0:int(0.075 * H), int(0.72 * W):W]
    crop = cv2.resize(crop, (0, 0), fx=0.75, fy=0.75, interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".png", crop)
    return buf.tobytes() if ok else None, crop.shape[1], crop.shape[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scans-dir", default=SCANS_DIR)
    ap.add_argument("--out", default=OUT_XLSX)
    ap.add_argument("--labels", default=LABELS_FILE or None,
                    help="optional csv/xlsx with columns patient_id,label")
    ap.add_argument("--gain-default", type=float, default=1.0)
    ap.add_argument("--no-recursive", action="store_true")
    ap.add_argument("--no-thumbs", action="store_true", help="do not embed header crops")
    a = ap.parse_args()

    if not a.scans_dir:
        print("No folder given (neither --scans-dir nor SCANS_DIR in the file).")
        try:
            a.scans_dir = input("Type or paste the folder that contains the scan images: ").strip().strip('"').strip("'")
        except EOFError:
            sys.exit("Run it like:  python 11_make_scans_xlsx.py --scans-dir \"E:\\ECG\\scans\"")
    root = Path(a.scans_dir)
    if not root.exists():
        sys.exit(f"folder not found: {root}\nCheck the path (use quotes if it contains spaces).")
    files = sorted(p for p in (root.iterdir() if a.no_recursive else root.rglob("*"))
                   if p.is_file() and p.suffix.lower() in EXTS)
    if not files:
        sys.exit("no images found")

    ext_labels = {}
    if a.labels:
        lb = pd.read_excel(a.labels) if a.labels.lower().endswith(".xlsx") else pd.read_csv(a.labels)
        lb["k"] = lb["patient_id"].astype(str).str.replace(r"^[Pp]", "", regex=True).str.strip()
        ext_labels = dict(zip(lb["k"], lb["label"].astype(str).str.upper().str.strip()))

    rows, checks, thumbs, sizes = [], [], {}, {}
    unparsed = []
    for p in files:
        pid, suf = parse_name(p.stem)
        if pid is None:
            unparsed.append(p.name)
            continue
        img, why = read_img(p)
        if img is None:
            checks.append(("ERROR", f"unreadable image: {p.name} -> {why}"))
            continue
        lab_folder = label_from_path(p, root)
        lab_ext = ext_labels.get(pid, "")
        if lab_folder == "CONFLICT":
            checks.append(("ERROR", f"{p.name}: folder path contains both LVOT and RVOT"))
            lab_folder = ""
        if lab_folder and lab_ext and lab_folder != lab_ext:
            checks.append(("ERROR", f"{p.name}: folder says {lab_folder} but labels file says {lab_ext}"))
        label = lab_folder or lab_ext
        src = "folder name" if lab_folder else ("labels file" if lab_ext else "MISSING")
        sid = p.stem
        rows.append({"scan_id": sid, "file": str(p.resolve()), "patient_id": f"P{pid}",
                     "file_suffix": suf, "label": label, "label_source": src,
                     "gain_mV_per_cm": a.gain_default, "gain_verified": "",
                     "image_size": f"{img.shape[1]}x{img.shape[0]}"})
        sizes[sid] = (img.shape[1], img.shape[0])
        if not a.no_thumbs:
            thumbs[sid] = header_crop_png(img)
    df = pd.DataFrame(rows)
    if df.empty:
        print(f"{len(files)} image files found in {root}, but none usable.")
        if unparsed:
            print(f"  {len(unparsed)} names without any number, e.g.: {unparsed[:5]}")
        for sev, msg in checks[:5]:
            print(f"  {sev}: {msg}")
        sys.exit(1)

    # ------------------------------------------------ checks
    if unparsed:
        checks.append(("WARN", f"{len(unparsed)} images skipped (name does not start with a patient number): "
                               f"{unparsed[:8]}"))
    dup = df[df.scan_id.duplicated(keep=False)]
    if len(dup):
        checks.append(("ERROR", f"duplicate scan_id (same file name in different folders): "
                                f"{sorted(dup.scan_id.unique())[:10]}. Rename them."))
    miss = df[df.label == ""]
    if len(miss):
        checks.append(("WARN", f"{len(miss)} scans / {miss.patient_id.nunique()} patients have NO label: "
                               f"fill the 'label' column (LVOT or RVOT). "
                               f"Patients: {sorted(miss.patient_id.unique())[:15]}"))
    lab_n = df[df.label != ""].groupby("patient_id").label.nunique()
    for pid in lab_n[lab_n > 1].index:
        checks.append(("ERROR", f"patient {pid} has scans with CONFLICTING labels"))
    sz = Counter(df.image_size)
    dom, dom_n = sz.most_common(1)[0]
    checks.append(("INFO", f"image sizes: {dict(sz)}"))
    if len(sz) > 1:
        odd = df[df.image_size != dom].scan_id.tolist()
        checks.append(("WARN", f"{len(odd)} scans differ from the dominant size {dom}: {odd[:10]}. "
                               f"Layout detection was tuned on 2576x1873 scans; inspect their QC images."))
    suf = Counter(df.file_suffix)
    checks.append(("INFO", f"file_suffix values (the part after the patient number): {dict(suf)}"))
    if len(suf) == 1:
        checks.append(("INFO", "The suffix is the same in every file, so it carries no information to "
                               "distinguish patients or scans. It is NOT used for patient identity."))
    else:
        checks.append(("WARN", "The suffix is not constant. If it means 'another recording of the same "
                               "patient' that is fine (all recordings keep the same patient_id and the "
                               "splitter keeps them together). Check what it means before you rely on it."))
    per_pat = df.groupby("patient_id").size()
    checks.append(("INFO", f"{len(df)} scans, {df.patient_id.nunique()} patients; scans per patient: "
                           f"{dict(Counter(per_pat))}"))
    checks.append(("WARN", "gain_mV_per_cm is the default for ALL rows until you verify it. Look at the "
                           "header crop in each row (e.g. '25mm/sec 1.0mV/cm' or '2.0mV/cm'), correct the "
                           "number, and type 'yes' in gain_verified. In many scans the top of the header is "
                           "cut off, so read the visible lower half carefully."))

    pat = (df.groupby("patient_id")
             .agg(n_scans=("scan_id", "size"),
                  labels=("label", lambda s: "/".join(sorted(set(x for x in s if x)))),
                  suffixes=("file_suffix", lambda s: ",".join(sorted(set(s)))))
             .reset_index())

    # ------------------------------------------------ write xlsx
    from openpyxl import Workbook
    from openpyxl.drawing.image import Image as XLImage
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.worksheet.datavalidation import DataValidation

    wb = Workbook()
    ws = wb.active
    ws.title = "scans"
    cols = ["scan_id", "file", "patient_id", "file_suffix", "label", "label_source",
            "gain_mV_per_cm", "gain_verified", "header_crop", "image_size"]
    ws.append(cols)
    for c in ws[1]:
        c.font = Font(bold=True, color="FFFFFF")
        c.fill = PatternFill("solid", fgColor="2F5597")
    widths = {"scan_id": 12, "file": 40, "patient_id": 11, "file_suffix": 11, "label": 9,
              "label_source": 13, "gain_mV_per_cm": 15, "gain_verified": 13, "header_crop": 36,
              "image_size": 11}
    for i, c in enumerate(cols, start=1):
        ws.column_dimensions[ws.cell(1, i).column_letter].width = widths[c]
    img_col = ws.cell(1, cols.index("header_crop") + 1).column_letter
    for r, row in enumerate(df.itertuples(index=False), start=2):
        for j, c in enumerate(cols, start=1):
            if c == "header_crop":
                continue
            ws.cell(r, j, getattr(row, c))
        ws.cell(r, cols.index("gain_mV_per_cm") + 1).fill = PatternFill("solid", fgColor="FFF2CC")
        if not thumbs:
            continue
        png, w, h = thumbs[row.scan_id]
        if png:
            im = XLImage(io.BytesIO(png))
            im.width, im.height = w, h
            ws.add_image(im, f"{img_col}{r}")
            ws.row_dimensions[r].height = max(30, h * 0.78)
        for j in range(1, len(cols) + 1):
            ws.cell(r, j).alignment = Alignment(vertical="center")
    n = len(df) + 1
    dv = DataValidation(type="list", formula1='"LVOT,RVOT"', allow_blank=True)
    dv2 = DataValidation(type="list", formula1='"yes,no"', allow_blank=True)
    ws.add_data_validation(dv)
    ws.add_data_validation(dv2)
    dv.add(f"{ws.cell(2, cols.index('label') + 1).column_letter}2:"
           f"{ws.cell(n, cols.index('label') + 1).column_letter}{n}")
    dv2.add(f"{ws.cell(2, cols.index('gain_verified') + 1).column_letter}2:"
            f"{ws.cell(n, cols.index('gain_verified') + 1).column_letter}{n}")
    ws.freeze_panes = "B2"

    wp = wb.create_sheet("patients")
    wp.append(list(pat.columns))
    for row in pat.itertuples(index=False):
        wp.append(list(row))
    wc = wb.create_sheet("checks")
    wc.append(["severity", "message"])
    for sev, msg in checks:
        wc.append([sev, msg])
        if sev == "ERROR":
            wc.cell(wc.max_row, 1).fill = PatternFill("solid", fgColor="F4B6B6")
        elif sev == "WARN":
            wc.cell(wc.max_row, 1).fill = PatternFill("solid", fgColor="FFE699")
    wc.column_dimensions["A"].width = 10
    wc.column_dimensions["B"].width = 140
    wr = wb.create_sheet("readme")
    for line in ["scans.xlsx - how to use",
                 "1. Sheet 'scans': check/fill label (dropdown), gain_mV_per_cm, gain_verified.",
                 "2. patient_id is the first number in the file name. file_suffix is informational only.",
                 "3. Do NOT put patient names or hospital numbers in any column.",
                 "4. Sheet 'checks' lists problems found automatically. Fix every ERROR.",
                 "5. Then run: python 10_validate_extraction.py --scans scans.xlsx --out validation_out",
                 "The header crop contains only the right-hand gain text, never the name."]:
        wr.append([line])
    wr.column_dimensions["A"].width = 110
    wb.save(a.out)
    df.drop(columns=["image_size"]).to_csv(Path(a.out).with_suffix(".csv"), index=False, encoding="utf-8")

    print(f"{len(df)} scans, {df.patient_id.nunique()} patients -> {Path(a.out).resolve()} (+ .csv)")
    print("labels:", dict(Counter(df.label.replace('', 'MISSING'))), "| suffixes:", dict(suf))
    for sev, msg in checks:
        if sev in ("ERROR", "WARN"):
            print(f"{sev}: {msg[:200]}")


if __name__ == "__main__":
    main()
