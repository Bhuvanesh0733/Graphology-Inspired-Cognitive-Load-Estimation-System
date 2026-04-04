"""
Flask Backend — Cognitive Load Estimation (Graphological Analysis)
==================================================================
Endpoints:
  GET  /            → Web UI
  POST /predict     → Upload image + task_type → JSON prediction
  GET  /history     → Last 20 predictions (SQLite)
  GET  /model-info  → Model metadata
"""

import os
import json
import sqlite3
import datetime
import joblib
import numpy as np
from flask import Flask, request, jsonify, render_template, g
from werkzeug.utils import secure_filename

from utils.feature_extractor import (
    extract_all_features, FEATURE_NAMES, FEATURE_GROUPS, HIGH_LOAD_DIRECTION
)

# ── Config ────────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = 'static/uploads'
ALLOWED_EXT = {'png', 'jpg', 'jpeg', 'bmp', 'tiff'}
MODEL_DIR   = 'model'
DATABASE    = 'data/predictions.db'
# ─────────────────────────────────────────────────────────────────────────────

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('data', exist_ok=True)


# ── Model loading ─────────────────────────────────────────────────────────────
def load_model():
    paths = {k: os.path.join(MODEL_DIR, v) for k, v in {
        'model':   'cognitive_load_model.pkl',
        'scaler':  'scaler.pkl',
        'le':      'label_encoder.pkl',
        'info':    'model_info.json',
    }.items()}
    if not os.path.exists(paths['model']):
        return None, None, None, None
    model  = joblib.load(paths['model'])
    scaler = joblib.load(paths['scaler'])
    le     = joblib.load(paths['le'])
    with open(paths['info']) as f:
        info = json.load(f)
    return model, scaler, le, info


MODEL, SCALER, LABEL_ENCODER, MODEL_INFO = load_model()


# ── Database ──────────────────────────────────────────────────────────────────
def get_db():
    db = getattr(g, '_db', None)
    if db is None:
        db = g._db = sqlite3.connect(DATABASE)
        db.row_factory = sqlite3.Row
    return db

@app.teardown_appcontext
def close_db(_):
    db = getattr(g, '_db', None)
    if db:
        db.close()

def init_db():
    with app.app_context():
        get_db().execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                filename   TEXT,
                task_type  TEXT,
                load_level TEXT,
                confidence REAL,
                status     TEXT,
                timestamp  TEXT,
                features   TEXT
            )
        ''')
        get_db().commit()


# ── Task-aware logic ──────────────────────────────────────────────────────────
EXPECTED = {
    'copying':         'LOW',
    'memory':          'MEDIUM',
    'problem_solving': 'HIGH',
}
LOAD_RANK = {'LOW': 0, 'MEDIUM': 1, 'HIGH': 2}

def task_status(predicted, task):
    exp  = EXPECTED.get(task, 'MEDIUM')
    diff = abs(LOAD_RANK.get(predicted, 1) - LOAD_RANK.get(exp, 1))
    if diff == 0:   return 'NORMAL',            exp
    if diff == 1:   return 'SLIGHTLY_ELEVATED', exp
    return             'ABNORMAL',              exp


# ── Graphological interpretation ───────────────────────────────────────────────
INSIGHTS = {
    "pressure_variance": {
        True:  "Stroke pressure is inconsistent — motor control under load",
        False: "Pressure is steady — confident pen strokes",
    },
    "slant_deviation": {
        True:  "Slant fluctuates significantly — divided attention pattern",
        False: "Consistent pen angle — controlled, focused writing",
    },
    "tremor_index": {
        True:  "Contour jaggedness detected — fine motor tremor present",
        False: "Smooth contours — no significant tremor",
    },
    "baseline_deviation": {
        True:  "Unsteady baseline — writing drifts vertically (fatigue marker)",
        False: "Stable baseline — writing follows a consistent line",
    },
    "pen_lift_fragmentation": {
        True:  "High fragmentation — frequent hesitations or pen lifts",
        False: "Fluid, connected strokes — low hesitation",
    },
    "letter_height_var": {
        True:  "Letter size varies significantly — size regulation impaired",
        False: "Consistent letter height — good size control",
    },
    "pixel_entropy": {
        True:  "Disordered pixel distribution — high image complexity",
        False: "Ordered image structure — clean, organised writing",
    },
    "zone_ratio": {
        True:  "Zone imbalance detected — upper/lower proportion disrupted",
        False: "Balanced writing zones — normal form production",
    },
}

def build_insights(features_dict, load_level):
    THRESHOLDS = {
        "pressure_variance":      1.2,
        "slant_deviation":        10.0,
        "tremor_index":           1.15,
        "baseline_deviation":     3.0,
        "pen_lift_fragmentation": 5.0,
        "letter_height_var":      4.0,
        "pixel_entropy":          4.5,
        "zone_ratio":             0.1,
    }
    result = []
    for feat, thresh in THRESHOLDS.items():
        val = features_dict.get(feat, 0)
        high = HIGH_LOAD_DIRECTION.get(feat, True)
        triggered = bool((val > thresh) if high else (val < thresh))
        text = INSIGHTS.get(feat, {}).get(triggered, "")
        if text:
            result.append({"feature": feat, "value": round(float(val), 4),
                           "triggered": triggered, "text": text})
    return result


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html',
                           model_ready=(MODEL is not None),
                           model_info=MODEL_INFO or {})


@app.route('/predict', methods=['POST'])
def predict():
    if MODEL is None:
        return jsonify({'error': 'Model not ready. Run python train_model.py first.'}), 503

    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded.'}), 400

    file = request.files['image']
    if not file.filename:
        return jsonify({'error': 'Empty filename.'}), 400

    ext = file.filename.rsplit('.', 1)[-1].lower()
    if ext not in ALLOWED_EXT:
        return jsonify({'error': f'File type .{ext} not supported.'}), 400

    task_type = request.form.get('task_type', 'copying')
    filename  = secure_filename(file.filename)
    filepath  = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    try:
        feat_dict, feat_vec = extract_all_features(filepath)

        feat_scaled = SCALER.transform([feat_vec])
        pred_enc    = MODEL.predict(feat_scaled)[0]
        proba       = MODEL.predict_proba(feat_scaled)[0]
        confidence  = float(np.max(proba))
        load_level  = LABEL_ENCODER.inverse_transform([pred_enc])[0]

        status, expected_load = task_status(load_level, task_type)

        class_probas = {
            cls: round(float(p), 3)
            for cls, p in zip(LABEL_ENCODER.classes_, proba)
        }

        # Group features for response
        grouped_features = {}
        for grp, names in FEATURE_GROUPS.items():
            grouped_features[grp] = {
                n: round(float(feat_dict[n]), 4) for n in names
            }

        insights = build_insights(feat_dict, load_level)

        result = {
            'load':               load_level,
            'confidence':         round(confidence, 3),
            'confidence_pct':     f"{round(confidence * 100, 1)}%",
            'status':             status,
            'expected_load':      expected_load,
            'task_type':          task_type,
            'class_probabilities': class_probas,
            'features':           {k: round(float(v), 4) for k, v in feat_dict.items()},
            'grouped_features':   grouped_features,
            'insights':           insights,
        }

        # Persist
        db = get_db()
        db.execute('''
            INSERT INTO predictions
            (filename, task_type, load_level, confidence, status, timestamp, features)
            VALUES (?,?,?,?,?,?,?)
        ''', (filename, task_type, load_level, confidence, status,
              datetime.datetime.now().isoformat(),
              json.dumps({k: round(float(v), 4) for k, v in feat_dict.items()})))
        db.commit()

        return jsonify(result)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500


@app.route('/history')
def history():
    rows = get_db().execute(
        'SELECT * FROM predictions ORDER BY id DESC LIMIT 20'
    ).fetchall()
    out = []
    for row in rows:
        r = dict(row)
        r['features'] = json.loads(r['features']) if r['features'] else {}
        out.append(r)
    return jsonify(out)


@app.route('/model-info')
def model_info():
    if MODEL_INFO is None:
        return jsonify({'error': 'No model found.'}), 404
    return jsonify(MODEL_INFO)


if __name__ == '__main__':
    init_db()
    print("\n🧠 Cognitive Load Estimation Server")
    print(f"   Model loaded : {MODEL is not None}")
    if MODEL_INFO:
        print(f"   Model        : {MODEL_INFO.get('model_name')}")
        print(f"   Accuracy     : {MODEL_INFO.get('accuracy')}")
        print(f"   Features     : {len(FEATURE_NAMES)} graphological")
    print("   URL          : http://localhost:5000\n")
    app.run(debug=True, host='0.0.0.0', port=5000)