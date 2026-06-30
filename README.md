# Graphology-Inspired Cognitive Load Estimation System

> Estimating mental effort from handwriting patterns using OpenCV-based graphological feature extraction and machine learning.

**Live Demo →** [https://graphology-inspired-cognitive-load.onrender.com/](https://graphology-inspired-cognitive-load.onrender.com/)

> Note: hosted on Render's free tier — the app sleeps after 15 minutes of inactivity, so the first request may take 30–50 seconds to wake up.

---

## What is this?

This project estimates a person's **cognitive load** — how hard their brain is working — purely from a handwriting image.

This is **not** stress or emotion detection. Cognitive load specifically measures mental/working-memory demand:

| Task | Cognitive Load |
|---|---|
| Copying a sentence | Low |
| Recalling something from memory | Medium |
| Solving a problem while writing | High |

The underlying idea comes from motor-control research: when the brain is under heavy cognitive load, fine motor signals to the hand degrade in measurable ways — strokes get shakier, pen pressure becomes inconsistent, slant angle fluctuates, letter spacing becomes erratic. Graphology gives a vocabulary to name and quantify these patterns; this project turns that vocabulary into computable features.

---

## How it works

```
Handwriting image
      │
      ▼
OpenCV preprocessing (grayscale → denoise → adaptive threshold)
      │
      ▼
15 graphological features extracted across 4 categories
      │
      ▼
StandardScaler → XGBoost / Random Forest classifier
      │
      ▼
Predicted load: LOW / MEDIUM / HIGH + confidence score
      │
      ▼
Task-aware logic compares prediction vs expected load → NORMAL / ABNORMAL
```

### The 15 graphological features

**Pressure patterns** — `pressure_mean`, `pressure_variance`, `ink_density`
Stroke thickness via distance transform; thinner/inconsistent strokes signal motor strain.

**Slant patterns** — `slant_mean_angle`, `slant_deviation`, `slant_skewness`
Pen tilt consistency across letters, fitted via `cv2.fitEllipse`; inconsistent slant signals divided attention.

**Spacing patterns** — `letter_spacing_mean`, `letter_spacing_var`, `baseline_deviation`
Gaps between letters and vertical drift of the writing baseline.

**Form & shape patterns** — `form_regularity`, `tremor_index`, `pen_lift_fragmentation`, `letter_height_var`, `pixel_entropy`, `zone_ratio`
Letter shape compactness, micro-tremor (perimeter vs convex hull ratio), pen-lift frequency, letter size consistency, image-wide pixel entropy, and upper/lower zone ink balance.

Each feature is computed using pure OpenCV/NumPy/SciPy operations on the binarized handwriting image — no deep learning, no black-box embeddings, every number is interpretable.

---

## Dataset

Trained on **115,318 word images** from the [IAM Handwriting Word Database](https://fki.tic.heia-fr.ch/databases/iam-handwriting-database).

IAM is built for handwriting *recognition*, not cognitive load, so it has no load labels. Labels here are assigned using a line-position proxy grounded in motor-fatigue literature (Rosenblum et al., 2010): writing further into a passage shows measurable degradation as fatigue accumulates.

| Line index (within a form) | Assigned label |
|---|---|
| 00 – 02 | LOW |
| 03 – 05 | MEDIUM |
| 06 + | HIGH |

This proxy-labeling is a deliberate, documented limitation — not a flaw — and is the basis for the project's main future-work direction (see below).

---

## Model performance

| Metric | Score |
|---|---|
| Model | XGBoost (beat Random Forest on test accuracy) |
| Test accuracy | 41.3% |
| Weighted F1 | 0.394 |
| Classes | LOW / MEDIUM / HIGH |
| Train / test split | 92,254 / 23,064 (stratified 80/20) |

41% on a 3-class problem is well above the 33% random baseline, and reflects genuine overlap between proxy-labeled classes rather than a modeling failure — see *Limitations* below.

---

## Tech stack

| Layer | Technology |
|---|---|
| Image processing | OpenCV |
| Numerical computing | NumPy, SciPy |
| Machine learning | scikit-learn, XGBoost |
| Data handling | Pandas |
| Visualization | Matplotlib, Seaborn |
| Backend | Flask, Gunicorn |
| Database | SQLite |
| Frontend | HTML5, CSS3, vanilla JavaScript |
| Deployment | Render |

---

## Project structure

```
cognitive_load_project/
│
├── app.py                       ← Flask server (predict, history, model-info routes)
├── train_model.py               ← Trains RF + XGBoost, saves best model
├── requirements.txt
├── Procfile                     ← Render deployment entrypoint
├── runtime.txt                  ← Pinned Python version
│
├── utils/
│   ├── feature_extractor.py     ← 15 graphological features (4 categories)
│   └── prepare_dataset.py       ← Scans IAM dataset → builds dataset.csv
│
├── data/
│   ├── dataset.csv              ← Generated feature dataset (not in repo)
│   └── labels/manual_labels.csv ← Optional manual label overrides
│
├── model/                       ← Trained model, scaler, encoder, evaluation plots
│
├── templates/index.html
└── static/
    ├── css/style.css
    └── js/main.js
```

---

## Running locally

```bash
git clone https://github.com/Bhuvanesh0733/Graphology-Inspired-Cognitive-Load-Estimation-System.git
cd Graphology-Inspired-Cognitive-Load-Estimation-System

pip install -r requirements.txt

# Place the IAM word dataset at iam_words/words/...
python utils/prepare_dataset.py     # builds data/dataset.csv
python train_model.py               # trains and saves the model
python app.py                       # → http://localhost:5000
```

---

## API

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Web UI |
| `/predict` | POST | Upload image + task_type → JSON prediction |
| `/history` | GET | Last 20 predictions |
| `/model-info` | GET | Model metadata |

`POST /predict` accepts `multipart/form-data`:
- `image` — PNG/JPG file
- `task_type` — `copying` \| `memory` \| `problem_solving`

---

## Limitations & future work

- Labels are a **proxy** (line position), not ground-truth cognitive load measured via validated instruments (e.g. NASA-TLX). A real follow-up study would collect handwriting under controlled, deliberately varied cognitive tasks with self-reported or physiologically validated load scores.
- IAM is offline, static handwriting — true cognitive load research also uses **stroke timing and pen-lift duration**, which require dynamic/online handwriting capture (e.g. a digital tablet), not scanned images.
- Current accuracy (41%) reflects label-proxy overlap, not a ceiling on the feature set; with validated labels, the same 15 features are expected to perform substantially better.
- Planned extension: CNN+LSTM model on sequential stroke data for writers who provide live pen input.

---

## References

Rosenblum, S., Parush, S., & Weiss, P. L. (2010). *Computerized temporal handwriting characteristics of proficient and non-proficient handwriters.* American Journal of Occupational Therapy.

IAM Handwriting Database — Institute of Computer Science and Applied Mathematics, University of Bern.

---

## License

This project is open for educational and research use. Dataset usage is subject to the IAM Handwriting Database's own license terms.
