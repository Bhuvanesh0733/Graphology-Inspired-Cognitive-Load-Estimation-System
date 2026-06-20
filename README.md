# Cognitive Load Estimation — Graphological Analysis

Estimates cognitive load (LOW / MEDIUM / HIGH) from handwriting images using
15 graphological features extracted with OpenCV, trained on the IAM word dataset.

---

## Project Structure

```
cognitive_load_project/
│
├── app.py                       ← Flask server  (run LAST)
├── train_model.py               ← Model training (run SECOND)
├── requirements.txt
│
├── iam_words/                   ← YOUR DATASET goes here
│   └── words/
│       ├── a01/
│       │   └── a01-000u/
│       │       ├── a01-000u-00-00.png
│       │       └── ...
│       ├── a02/ ...
│       └── b01/ ...
│
├── utils/
│   ├── feature_extractor.py     ← 15 graphological features (4 groups)
│   └── prepare_dataset.py       ← Scans IAM folder → data/dataset.csv
│
├── data/
│   ├── dataset.csv              ← Auto-generated
│   ├── predictions.db           ← SQLite (auto-created by Flask)
│   └── labels/
│       └── manual_labels.csv    ← Optional label overrides
│
├── model/                       ← Auto-generated after training
├── templates/index.html
└── static/css/ + js/
```

---

## 3-Step Setup

### 1. Install
```bash
pip install -r requirements.txt
```

### 2. Build dataset
```bash
python utils/prepare_dataset.py
```
Labels assigned by line position in filename (`a01-000u-LINE-WORD.png`):
- Lines 00-02 → **LOW**
- Lines 03-05 → **MEDIUM**
- Lines 06+   → **HIGH**

Override any label in `data/labels/manual_labels.csv` (format: `filename,label`)

### 3. Train
```bash
python train_model.py
```

### 4. Run web app
```bash
python app.py
# → http://localhost:5000
```

---

## The 15 Graphological Features

**Pressure:** pressure_mean, pressure_variance, ink_density

**Slant:** slant_mean_angle, slant_deviation, slant_skewness

**Spacing:** letter_spacing_mean, letter_spacing_var, baseline_deviation

**Form/Shape:** form_regularity, tremor_index, pen_lift_fragmentation,
               letter_height_var, pixel_entropy, zone_ratio

---
