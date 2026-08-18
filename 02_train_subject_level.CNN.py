"""
02_train_subject_level.py
--------------------------
Loads the preprocessed LVOT/RVOT beat images produced by
01_preprocess_beats.py, and trains/evaluates the CNN with a
SUBJECT-LEVEL (patient-level) split instead of a beat-level split.


"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from sklearn.metrics import (
    confusion_matrix, roc_curve, roc_auc_score,
    f1_score, precision_score, recall_score
)

import tensorflow as tf

from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import AdamW

# ------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------
PROCESSED_DIR = Path(r"E:\ECG\GithubCodes\Processed")   # OUT_DIR from script 01
MANIFEST_CSV = PROCESSED_DIR / "beats_manifest.csv"
LVOT_DIR = PROCESSED_DIR / "LVOT_Final"
RVOT_DIR = PROCESSED_DIR / "RVOT_Final"

IMAGE_SIZE = (128, 128)
LVOT_LABEL = 0
RVOT_LABEL = 1

TEST_SIZE = 0.2        # patient-level holdout fraction
K_FOLDS = 5
RANDOM_STATE = 42


# ------------------------------------------------------------------
# 1) Load data (images, labels, patient_id) using the manifest
#    written by 01_preprocess_beats.py
# ------------------------------------------------------------------
def load_dataset():
    manifest = pd.read_csv(MANIFEST_CSV)
    manifest = manifest[manifest["status"] == "kept"].reset_index(drop=True)

    images, labels, patient_ids, focus_ids = [], [], [], []

    for _, row in manifest.iterrows():
        class_name = row["class"]
        folder = LVOT_DIR if class_name == "LVOT" else RVOT_DIR
        img_path = folder / row["new_filename"]
        if not img_path.exists():
            continue

        img = load_img(img_path, target_size=IMAGE_SIZE, color_mode="grayscale")
        img_array = img_to_array(img) / 255.0

        images.append(img_array)
        labels.append(LVOT_LABEL if class_name == "LVOT" else RVOT_LABEL)
        patient_ids.append(str(row["patient_id"]))
        focus_ids.append(str(row["focus_id"]))

    images = np.array(images)
    labels = np.array(labels)
    patient_ids = np.array(patient_ids)
    focus_ids = np.array(focus_ids)

    return images, labels, patient_ids, focus_ids, manifest


# ------------------------------------------------------------------
# 2) Reviewer-facing summary report
# ------------------------------------------------------------------
def print_reviewer_report(labels, patient_ids, focus_ids, manifest):
    n_beats = len(labels)
    n_patients = len(set(patient_ids))
    n_foci = len(set(focus_ids))
    n_excluded = (manifest["status"] == "excluded").sum() if "status" in manifest.columns else "see script 01 log"

    beats_per_patient = pd.Series(patient_ids).value_counts()

    print("\n================ REVIEWER REPORT ================")
    print(f"Retained beats used for modeling:      {n_beats}")
    print(f"Unique patients (subjects):            {n_patients}")
    print(f"Unique active foci:                    {n_foci}")
    print(f"Beats per patient - min/median/max:    "
          f"{beats_per_patient.min()} / {beats_per_patient.median():.1f} / {beats_per_patient.max()}")
    print(f"Patients contributing >1 beat:         {(beats_per_patient > 1).sum()} "
          f"({(beats_per_patient > 1).mean()*100:.1f}% of patients)")
    print("Train/validation/test split performed at: PATIENT LEVEL "
          "(GroupShuffleSplit + StratifiedGroupKFold, group = patient_id)")
    print("===================================================\n")


# ------------------------------------------------------------------
# 3) Model definition (unchanged architecture from original notebook)
# ------------------------------------------------------------------
def center_normalize(x):
    return (x - tf.reduce_mean(x)) / tf.math.reduce_std(x)


def create_model():
    model = models.Sequential([
        layers.Lambda(center_normalize, input_shape=(128, 128, 1)),

        layers.Conv2D(16, (3, 9), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        layers.Flatten(),
        layers.Dense(1000, activation='relu', kernel_regularizer=l2(0.001)),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer=AdamW(learning_rate=0.0001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


# ------------------------------------------------------------------
# 4) Main pipeline: preprocessing already done (script 01) -> now split
# ------------------------------------------------------------------
def main():
    images, labels, patient_ids, focus_ids, manifest = load_dataset()
    print(f"Loaded {images.shape[0]} preprocessed beats, image shape {images.shape[1:]}")

    print_reviewer_report(labels, patient_ids, focus_ids, manifest)

    # ---- Patient-level Train+Val / Test split -------------------------
    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    train_val_idx, test_idx = next(gss.split(images, labels, groups=patient_ids))

    X_train_val, X_test = images[train_val_idx], images[test_idx]
    y_train_val, y_test = labels[train_val_idx], labels[test_idx]
    groups_train_val = patient_ids[train_val_idx]

    assert set(patient_ids[train_val_idx]).isdisjoint(set(patient_ids[test_idx])), \
        "Patient leakage between train+val and test!"

    print(f"Train+Val: {len(X_train_val)} beats from {len(set(groups_train_val))} patients")
    print(f"Test:      {len(X_test)} beats from {len(set(patient_ids[test_idx]))} patients")

    # ---- Data augmentation (unchanged) ---------------------------------
    data_gen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.15,
        height_shift_range=0.15,
        zoom_range=0.15,
        fill_mode='nearest',
        preprocessing_function=lambda x: tf.image.random_jpeg_quality(x, 80, 100)
    )

    # ---- Stratified Group K-Fold CV (group = patient_id) ---------------
    sgkf = StratifiedGroupKFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    fold_accuracies = []
    fold_history = []
    best_model = None
    best_accuracy = 0.0

    for fold, (train_idx, val_idx) in enumerate(
            sgkf.split(X_train_val, y_train_val, groups=groups_train_val)):

        print(f"\n Training Fold {fold + 1}/{K_FOLDS}")

        X_train, X_val = X_train_val[train_idx], X_train_val[val_idx]
        y_train, y_val = y_train_val[train_idx], y_train_val[val_idx]

        # sanity check: no patient overlap between train and val within this fold
        assert set(groups_train_val[train_idx]).isdisjoint(set(groups_train_val[val_idx])), \
            f"Patient leakage between train/val in fold {fold + 1}!"

        train_generator = data_gen.flow(X_train, y_train, batch_size=16, shuffle=True)

        model = create_model()
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)

        history = model.fit(
            train_generator,
            validation_data=(X_val, y_val),
            epochs=150,
            callbacks=[early_stopping, lr_scheduler],
            verbose=1
        )

        _, accuracy = model.evaluate(X_val, y_val, verbose=0)
        fold_accuracies.append(accuracy)
        fold_history.append(history)
        print(f"Fold {fold + 1} Accuracy: {accuracy:.4f}")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model

    print("\nFold accuracies:", fold_accuracies)
    best_fold_idx = int(np.argmax(fold_accuracies))
    print("Best fold index:", best_fold_idx)
    history = fold_history[best_fold_idx]

    # ---- Plots ----------------------------------------------------------
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(PROCESSED_DIR / "training_curves.png", dpi=150)
    plt.show()

    # ---- Final evaluation on the untouched, patient-level TEST set -----
    y_pred = best_model.predict(X_test)
    y_pred_labels = (y_pred > 0.5).astype(int)

    auc = roc_auc_score(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_pred)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve (patient-level test set)')
    plt.legend(loc='lower right')
    plt.savefig(PROCESSED_DIR / "roc_curve.png", dpi=150)
    plt.show()

    cm = confusion_matrix(y_test, y_pred_labels)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['LVOT', 'RVOT'], yticklabels=['LVOT', 'RVOT'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix (patient-level test set)')
    plt.savefig(PROCESSED_DIR / "confusion_matrix.png", dpi=150)
    plt.show()

    TP, TN = cm[1, 1], cm[0, 0]
    FP, FN = cm[0, 1], cm[1, 0]

    accuracy = (TP + TN) / (TP + TN + FP + FN)
    sensitivity = TP / (TP + FN)
    specificity = TN / (TN + FP)
    f1 = f1_score(y_test, y_pred_labels)
    precision = precision_score(y_test, y_pred_labels)
    recall = recall_score(y_test, y_pred_labels)

    print("\n================ FINAL TEST RESULTS (patient-level, unseen patients) ================")
    print(f"Test patients: {len(set(patient_ids[test_idx]))}   Test beats: {len(X_test)}")
    print(f"Accuracy:    {accuracy:.4f}")
    print(f"Sensitivity: {sensitivity:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print(f"Precision:   {precision:.4f}")
    print(f"Recall:      {recall:.4f}")
    print(f"F1-score:    {f1:.4f}")
    print(f"AUC:         {auc:.4f}")
    print("========================================================================================")


if __name__ == "__main__":
    main()
