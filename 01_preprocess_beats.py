"""
01_preprocess_beats.py
-----------------------
Preprocessing of LVOT / RVOT ECG beat images.

"""

import os
import re
import glob
import csv
from pathlib import Path

import cv2
import numpy as np

# ------------------------------------------------------------------
# CONFIG - edit these paths/settings for your machine
# ------------------------------------------------------------------
RAW_LVOT_DIR = r"E:\ECG\GithubCodes\LVOT"      # original raw beat images (LVOT)
RAW_RVOT_DIR = r"E:\ECG\GithubCodes\RVOT"      # original raw beat images (RVOT)

OUT_DIR = Path(r"E:\ECG\GithubCodes\Processed")       # all outputs go here
OUT_LVOT = OUT_DIR / "LVOT_Final"
OUT_RVOT = OUT_DIR / "RVOT_Final"
EXCLUSION_LOG = OUT_DIR / "excluded_beats_log.csv"
MANIFEST_CSV = OUT_DIR / "beats_manifest.csv"      # filename -> patient_id, focus_id, class, kept/excluded

TARGET_SIZE = (128, 128)
MIN_WIDTH_LVOT = 100
MIN_WIDTH_RVOT = 90


# ------------------------------------------------------------------
# >>> EDIT THIS FUNCTION <<<
# Extract a stable PATIENT/SUBJECT identifier from the ORIGINAL raw
# filename (before any renaming). This is the single most important
# piece needed to answer the reviewer's comment about repeated
# measures - if this is wrong, the subject-level split downstream is
# meaningless.
#
# Send me 3-5 example raw filenames from LVOT_Beats / RVOT_Beats and
# I will finalize this regex precisely for you.
#
# Guesses to try, uncomment/adapt whichever matches your naming:
# ------------------------------------------------------------------
def extract_patient_id(raw_filename: str) -> str:
    """
    Return a patient/subject ID string parsed from the raw filename.

    Example patterns seen in similar ECG-beat datasets (ADAPT AS NEEDED):
      "Patient12_Abnormal_lead_aVR.jpg"      -> "12"
      "P012_beat3_Abnormal_aVR.jpg"          -> "012"
      "12_034_Abnormal_aVR.jpg"              -> "12"   (first number = patient)
    """
    name = os.path.basename(raw_filename)

    # --- Option A: filename starts with "Patient" / "P" + digits ---
    m = re.search(r'(?:patient|pt|p)[_\-]?0*(\d+)', name, flags=re.IGNORECASE)
    if m:
        return m.group(1)

    # --- Option B (fallback): first number found anywhere in filename ---
    m = re.search(r'(\d+)', name)
    if m:
        return m.group(1)

    raise ValueError(
        f"Could not extract a patient ID from filename: {name}. "
        f"Please edit extract_patient_id() in this script."
    )


def extract_focus_id(raw_filename: str) -> str:
    """
    OPTIONAL: if your raw filenames also encode a distinct 'active focus'
    ID (a patient could in principle have more than one focus/ablation
    site), extract it here. If foci == patients in your dataset, just
    return extract_patient_id(raw_filename) here (default below).
    """
    return extract_patient_id(raw_filename)


# Standard 12-lead ECG names, ordered longest-first so e.g. "aVR" is matched
# before a bare "V" or "R" could accidentally match.
_ECG_LEADS = [
    "aVR", "aVL", "aVF",
    "V1", "V2", "V3", "V4", "V5", "V6",
    "I", "II", "III",
]
_LEAD_PATTERN = re.compile(
    r'(?<![a-zA-Z0-9])(' + '|'.join(re.escape(l) for l in _ECG_LEADS) + r')(?![a-zA-Z0-9])',
    flags=re.IGNORECASE
)


def extract_lead_name(raw_filename: str) -> str:
    """
    Extract the ECG lead name (e.g. 'aVR', 'V1', 'II') from the raw
    filename, matched against the standard 12-lead names so it isn't
    confused with the patient ID or other digits in the filename.
    Falls back to the last underscore-separated token if no standard
    lead name is found, or 'NA' if nothing usable is found.
    """
    name = os.path.basename(raw_filename)
    match = _LEAD_PATTERN.search(name)
    if match:
        lead = match.group(1)
        # Normalize casing (aVR/aVL/aVF -> lowercase 'a' + uppercase rest)
        if lead[0].lower() == 'a' and len(lead) == 3:
            return 'a' + lead[1:].upper()
        return lead.upper()

    # Fallback: last underscore-separated token before the extension
    fallback = re.search(r'_([a-zA-Z0-9]+)\.(jpg|jpeg|png)$', name, flags=re.IGNORECASE)
    if fallback:
        return fallback.group(1)

    return "NA"


# ------------------------------------------------------------------
# Preprocessing pipeline (same filters as the original notebook:
# bilateral filter -> Otsu threshold -> crop -> resize)
# ------------------------------------------------------------------
def preprocess_one_image(image_path: str, min_width: int):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None, "unreadable"

    if image.shape[1] < min_width:
        return None, f"width<{min_width}"

    bilateral_filtered = cv2.bilateralFilter(image, d=10, sigmaColor=50, sigmaSpace=50)
    _, otsu_threshold = cv2.threshold(
        bilateral_filtered, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    otsu_threshold = (otsu_threshold > 0).astype(np.uint8) * 255

    # Crop top/bottom, keep left 200 px (same as original notebook)
    cropped = otsu_threshold[50:-50, 0:200]
    if cropped.size == 0:
        return None, "empty_after_crop"

    resized = cv2.resize(cropped, TARGET_SIZE, interpolation=cv2.INTER_AREA)
    return resized, None


def process_class(raw_dir: str, out_dir: Path, class_name: str, min_width: int,
                   manifest_rows: list, exclusion_rows: list):
    os.makedirs(out_dir, exist_ok=True)

    raw_files = [
        f for f in os.listdir(raw_dir)
        if "Abnormal" in f and f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]
    print(f"[{class_name}] {len(raw_files)} raw candidate files found in {raw_dir}")

    counter = 0
    for raw_name in raw_files:
        raw_path = os.path.join(raw_dir, raw_name)

        try:
            patient_id = extract_patient_id(raw_name)
            focus_id = extract_focus_id(raw_name)
        except ValueError as e:
            exclusion_rows.append([class_name, raw_name, "no_patient_id"])
            print(f"  SKIP (no patient id): {raw_name}")
            continue

        lead = extract_lead_name(raw_name)

        processed_img, reject_reason = preprocess_one_image(raw_path, min_width)

        counter += 1
        # New filename ENCODES class + patient_id + focus_id + running beat index + lead
        new_name = f"{class_name}_P{patient_id}_F{focus_id}_{counter:04d}_{lead}.png"

        if reject_reason is not None:
            exclusion_rows.append([class_name, raw_name, reject_reason])
            manifest_rows.append([new_name, class_name, patient_id, focus_id, lead, "excluded"])
            print(f"  SKIP ({reject_reason}): {raw_name}")
            continue

        save_path = out_dir / new_name
        cv2.imwrite(str(save_path), processed_img, [cv2.IMWRITE_PNG_COMPRESSION, 9])
        manifest_rows.append([new_name, class_name, patient_id, focus_id, lead, "kept"])

    return counter


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    manifest_rows = []
    exclusion_rows = []

    process_class(RAW_LVOT_DIR, OUT_LVOT, "LVOT", MIN_WIDTH_LVOT, manifest_rows, exclusion_rows)
    process_class(RAW_RVOT_DIR, OUT_RVOT, "RVOT", MIN_WIDTH_RVOT, manifest_rows, exclusion_rows)

    # Write manifest (every raw beat, kept or excluded, with patient/focus id)
    with open(MANIFEST_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["new_filename", "class", "patient_id", "focus_id", "lead", "status"])
        writer.writerows(manifest_rows)

    # Write exclusion log
    with open(EXCLUSION_LOG, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["class", "raw_filename", "reason"])
        writer.writerows(exclusion_rows)

    kept = [r for r in manifest_rows if r[5] == "kept"]
    excluded = [r for r in manifest_rows if r[5] == "excluded"]
    n_patients = len(set(r[2] for r in kept))
    n_foci = len(set(r[3] for r in kept))
    n_leads = len(set(r[4] for r in kept))

    print("\n================ SUMMARY (for the rebuttal letter) ================")
    print(f"Total raw beats found (Abnormal, LVOT+RVOT): {len(manifest_rows)}")
    print(f"Beats KEPT after preprocessing:               {len(kept)}")
    print(f"Beats EXCLUDED during preprocessing:           {len(excluded)}")
    print(f"Unique patients among kept beats:              {n_patients}")
    print(f"Unique active foci among kept beats:           {n_foci}")
    print(f"Unique ECG leads represented among kept beats: {n_leads}")
    print(f"Manifest written to:   {MANIFEST_CSV}")
    print(f"Exclusion log written: {EXCLUSION_LOG}")
    print("=====================================================================")


if __name__ == "__main__":
    main()
