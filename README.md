# Scanned ECG-Based Deep Learning for Localizing the Origin of Premature Ventricular Complexes

**A research pipeline for distinguishing right ventricular outflow tract (RVOT) from left ventricular outflow tract (LVOT) premature ventricular complex (PVC) origins using scanned 12-lead ECG recordings.**

<p align="center">
  <img src="graphical_abstract.svg" alt="Graphical abstract: scanned ECG-based deep learning for PVC-origin localization" width="100%">
</p>

## Overview

This repository contains the code for a study evaluating deep-learning methods for patient-level discrimination of RVOT- versus LVOT-origin PVCs from digitized paper-based ECG recordings.

The workflow covers scanned ECG processing, PVC and preceding sinus-beat window extraction, extraction-quality assessment, event-level feature construction, and patient-level model evaluation. Three model architectures are evaluated:

- **Custom CNN** — a two-dimensional convolutional neural network operating on ECG image crops.
- **MobileNetV2** — an image-based transfer-learning model.
- **Wave1D** — a one-dimensional CNN operating on reconstructed waveform envelopes.

The main comparison evaluates focused precordial input (V1–V3) against the complete 12-lead representation. The primary statistical unit is the patient, not the individual lead crop or PVC event.

This is a research proof of concept. It is not a medical device and is not intended to replace expert ECG interpretation, electrophysiological evaluation, intracardiac mapping, or definitive procedural localization.

## Study cohort

- **Initial cohort:** 59 adult patients with PVCs.
- **Final modeling cohort:** 54 patients after exclusion of five recordings that did not meet lead availability/quality requirements.
- **Final classes:** 21 LVOT and 33 RVOT patients.
- **Retained PVC events:** 75 unique events (24 LVOT-associated and 51 RVOT-associated).
- **Study setting:** Imam Reza Hospital, Mashhad, Iran.
- **Study period:** 2020–2024.
- **Reference standard:** 3D electroanatomical mapping and confirmation of the successful ablation site.

PVC events are nested within patients. They are therefore not treated as statistically independent observations in the patient-level evaluation.

## Graphical abstract

The graphical abstract is included as `graphical_abstract.svg`. GitHub renders this file directly in the README.

## Current analysis pipeline

The scripts below describe the current workflow in the repository. The earlier scripts `01_preprocess_beats.py`, `02_train_subject_level.CNN.py`, and `03_train_subject_level_mobilenetv2.py` are **legacy implementations** and are not part of the analysis reported in the current manuscript.

| Script | Role |
|---|---|
| `11_make_scans_xlsx.py` | Builds a scan inventory workbook from a local directory of ECG scans. It records scan paths, patient keys, labels when available, and gain information. |
| `08_extract_pvc_sinus_from_scan.py` | Processes one scanned 12-lead ECG. Extracts PVC and immediately preceding sinus-beat windows, creates scale-preserving image crops, and exports beat-level measurements and reconstructed waveform envelopes. |
| `09_batch_extract.py` | Runs the single-scan extraction procedure over a scan list and writes a combined beat manifest. |
| `10_validate_extraction.py` | Runs extraction and quality checks across scans; evaluates agreement with cardiologist-drawn windows, lead reliability, timing plausibility, and scan-level quality flags. |
| `12_build_event_features.py` | Builds event-level and patient-level feature tables from the validated extraction output. |
| `14_train_deep_patient_level.py` | Trains and evaluates the three models using the focused V1–V3 representation. |
| `14_train_deep_patient_level_12lead.py` | Runs the corresponding model pipeline with all 12 leads or a user-selected lead subset. |
| `14_1_lead_ablation.py` | Retrains models under lead-removal conditions to assess the contribution of V1, V2, and V3. |

### Processing details

The extraction script is designed for a specific 12-lead printout layout: three rows by four columns, plus a lead-II rhythm strip. Each panel represents a 2.5-second interval. Cardiologist-drawn colored annotation lines are used when available; otherwise, an automatic beat detector is applied. Quality-control outputs should always be reviewed.

The scan inventory must contain the fields required by the validation pipeline:

- `file`
- `patient_id`
- `gain_mV_per_cm`

Optional fields include `scan_id`, `label`, `file_suffix` or `recording`, and `gain_verified`. The gain value must be checked against the original ECG printout. Incorrect gain information changes amplitude measurements.

The feature-building script produces both event-level and patient-level tables. The patient-level table is used to restrict the modeling cohort consistently across analyses.

### Model inputs and evaluation

- **V1–V3 image input:** three scale-preserving PVC image channels.
- **V1–V3 waveform input:** upper and lower waveform envelopes for each of the three precordial leads (six channels).
- **12-lead image input:** image crops from all 12 leads.
- **12-lead waveform input:** upper and lower envelopes from all 12 leads (24 channels).
- **Architectures:** Custom CNN, MobileNetV2, and Wave1D.
- **Validation:** repeated stratified patient-level five-fold cross-validation (three repetitions by default in the scripts).
- **Aggregation:** event-level predictions are aggregated to patient-level scores.
- **Uncertainty:** patient-level bootstrap confidence intervals are used for reported performance estimates.
- **Interpretability:** Grad-CAM and lead-occlusion analyses are implemented; lead ablation retrains models after removing individual leads.

All PVC events from a patient must remain within the same fold. This patient-level partitioning is essential to avoid information leakage between training and evaluation data.

## Main results reported in the manuscript

### Patient-level discrimination

| Input | Model | AUC | 95% CI |
|---|---|---:|---:|
| V1–V3 | **Custom CNN** | **0.833** | 0.721–0.931 |
| V1–V3 | Wave1D | 0.812 | 0.685–0.926 |
| V1–V3 | MobileNetV2 | 0.812 | 0.675–0.927 |
| All 12 leads | Custom CNN | 0.795 | — |
| All 12 leads | Wave1D | 0.713 | — |
| All 12 leads | MobileNetV2 | 0.530 | — |

The V1–V3 configuration had numerically higher AUC values than the 12-lead configuration for all three architectures. These results do not establish that the remaining nine leads are clinically unnecessary or that a three-lead model is superior in other populations.

### Three-lead calibration

| Model | Brier score |
|---|---:|
| Custom CNN | 0.166 |
| Wave1D | 0.174 |
| MobileNetV2 | 0.170 |
| Prevalence-based reference | 0.238 |

Calibration and decision-curve analyses are exploratory and should not be interpreted as evidence of established clinical utility.

### Lead-ablation finding

In the V1–V3 analysis, removing V3 reduced Wave1D AUC from 0.812 to 0.606. The manuscript reports a paired AUC difference of −0.206 (95% CI −0.378 to −0.035). Lead contributions differed by architecture.

## Installation

The current `requirements.txt` lists NumPy, pandas, Matplotlib, Seaborn, OpenCV, scikit-learn, and TensorFlow. The current processing scripts also import **Pillow** and **SciPy**, and the Excel workflow requires **openpyxl**. These packages must be installed in the execution environment.

```bash
git clone https://github.com/AfroozArzehgar/Scanned-ECG-PVC-Classifier.git
cd Scanned-ECG-PVC-Classifier

python -m venv .venv

# Linux / macOS
source .venv/bin/activate

# Windows PowerShell
# .venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt
pip install Pillow scipy openpyxl
```

The repository does not currently pin a tested Python/TensorFlow version combination. For reproducible execution, record the exact Python version, TensorFlow version, operating system, and hardware used for each final run.

## Running the pipeline with authorized local data

**The repository does not distribute clinical ECG scans or patient-level datasets.** The commands below are for users who independently hold the necessary institutional and ethics authorization to process the source data.

### 1. Create a scan inventory

```bash
python 11_make_scans_xlsx.py \
  --scans-dir "/path/to/authorized/ecg_scans" \
  --out scans.xlsx \
  --no-thumbs
```

The workbook contains patient keys, scan paths, labels (if supplied), and gain information. Keep this workbook local: it contains patient-level metadata and file paths.

### 2. Validate and extract ECG windows

```bash
python 10_validate_extraction.py \
  --scans scans.xlsx \
  --out validation_out
```

The output includes scan-level extraction information, quality-control images, annotation-agreement results, and a validation report. Review the QC images and report before using the extracted data.

`09_batch_extract.py` is also available for batch extraction and manifest generation:

```bash
python 09_batch_extract.py scans.xlsx extraction_out
```

### 3. Build event- and patient-level features

```bash
python 12_build_event_features.py \
  --run-dir validation_out \
  --out features_out \
  --min-leads 9
```

Optional quality filtering:

```bash
python 12_build_event_features.py \
  --run-dir validation_out \
  --out features_out \
  --min-leads 9 \
  --drop-clipped
```

This step writes `event_features.csv` and `patient_features.csv`. These files contain derived patient/event-level information and must remain in the approved local environment unless data sharing is explicitly authorized.

### 4. Train and evaluate the V1–V3 models

```bash
python 14_train_deep_patient_level.py \
  --run-dir validation_out \
  --restrict-to features_out/patient_features.csv \
  --out deep_out
```

### 5. Run the 12-lead comparison

```bash
python 14_train_deep_patient_level_12lead.py \
  --run-dir validation_out \
  --restrict-to features_out/patient_features.csv \
  --out deep12_out \
  --leads all
```

The 12-lead script also supports a synthetic self-test that does not require clinical data:

```bash
python 14_train_deep_patient_level_12lead.py --selftest
```

The self-test uses generated synthetic data to exercise the code path. It is not a substitute for the clinical dataset and cannot reproduce the manuscript's clinical performance results.

### 6. Run the V1–V3 lead-ablation analysis

```bash
python 14_1_lead_ablation.py \
  --run-dir validation_out \
  --restrict-to features_out/patient_features.csv \
  --out ablation_out
```

The lead-ablation script imports the current three-lead training implementation. Keep `14_1_lead_ablation.py` and `14_train_deep_patient_level.py` in the same directory.

## Data availability, privacy, and ethical restrictions

**No patient-level clinical data are intentionally distributed in this repository.**

The clinical ECG scans were obtained from human participants under institutional ethical approval. Public redistribution of the source scans, derived beat crops, waveform arrays, patient-level feature tables, or other participant-level records requires explicit authorization from the responsible institution and ethics committee. De-identification alone does not automatically establish permission for public release.

Accordingly:

- Do not upload original ECG scans, screenshots, beat crops, waveform arrays, or model-ready patient-level tables.
- Do not upload `scans.xlsx`, `patient_features.csv`, `event_features.csv`, `beats_manifest_all.csv`, `beats_extracted.csv`, `waveforms.npz`, or validation/QC output folders.
- Do not assume that removing names or medical-record numbers is sufficient. Scan identifiers, dates, filenames, rare clinical patterns, and linked metadata may still create re-identification risks.
- Do not upload trained weights/checkpoints derived from the clinical cohort unless the institution has explicitly reviewed and approved their release.
- Use fully synthetic examples for public demonstrations. Clearly label them as synthetic and never present synthetic data as clinical observations.
- Keep all authorized source data and intermediate outputs in institutionally approved storage with access controls.

If the institution later authorizes controlled data access, document the approval process, data-use agreement, permitted users, and request procedure here. Until then, this repository should remain a **code-only research repository**.

### Recommended data-availability statement for the manuscript

> The code used for ECG preprocessing, beat extraction, feature construction, and model evaluation is publicly available at https://github.com/AfroozArzehgar/Scanned-ECG-PVC-Classifier. Individual-level clinical ECG recordings and derived participant-level datasets are not publicly available because their dissemination is subject to institutional and ethics-related restrictions. No participant-level clinical data are distributed through the repository.

The manuscript's current data-availability statement should be revised if it still states that de-identified data are publicly available.

## Ethics

The study was conducted in accordance with the Declaration of Helsinki and approved by the Ethics Committee of Imam Reza Hospital, Mashhad University of Medical Sciences, Mashhad, Iran (Approval No. **IR.MUMS.REC.1403.307**). Written informed consent was obtained from all participants.

## Limitations

- Single-center cohort with 54 patients in the final modeling dataset.
- Internal repeated cross-validation; no independent external validation cohort.
- Binary classification restricted to RVOT versus LVOT origins.
- The analysis uses scanned ECGs and does not establish equivalence with native digital ECG waveforms.
- Extraction and beat annotation depend on scan layout, image quality, gain information, and annotation quality.
- Calibration, decision-curve, and interpretability analyses are exploratory.
- The study did not establish clinical utility or demonstrate improved ablation outcomes.

External multicenter and prospective validation is required before any clinical application can be considered.

## Citation

If you use this code, please cite the associated manuscript:

```bibtex
@article{davarinia2026scannedecg,
  title   = {Scanned Electrocardiogram-Based Deep Learning for Localizing the Origin of Premature Ventricular Complexes},
  author  = {Davarinia, Fatemeh and Arzehgar, Afrooz and Rahimpour, Feisal and Ramezaninezhad, Davood},
  year    = {2026},
  note    = {Update with journal, volume, pages, and DOI when available}
}
```

## Authors and contributions

- **Fatemeh Davarinia:** Conceptualization, methodology, data analysis, and manuscript drafting.
- **Afrooz Arzehgar:** Data preprocessing and model implementation.
- **Feisal Rahimpour and Davood Ramezaninezhad:** Clinical data acquisition, patient evaluation, and electrophysiological validation.

All authors reviewed, edited, and approved the manuscript.

## Contact

- Fatemeh Davarinia — f_davarinia@semnan.ac.ir
- Afrooz Arzehgar — arzegara4011@mums.ac.ir

## License

A license file is not currently visible in the repository. No open-source license should be claimed until the authors have selected and added an appropriate `LICENSE` file and confirmed that they have the right to license the code.

---

**Research-use disclaimer:** This software is provided for research and educational purposes only. It has not been validated as a clinical device and must not be used as the sole basis for diagnosis, treatment, electrophysiological localization, or catheter-ablation decisions.
