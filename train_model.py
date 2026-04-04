"""
Model Training — Cognitive Load Estimation (Graphological Features)
====================================================================
Trains Random Forest + XGBoost on 15 graphological features.
Outputs:
  model/cognitive_load_model.pkl
  model/scaler.pkl
  model/label_encoder.pkl
  model/model_info.json
  model/confusion_matrix.png
  model/feature_importance.png
  model/feature_group_importance.png
"""

import os
import sys
import json
import joblib
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.inspection import permutation_importance

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("XGBoost not installed — GradientBoosting will be used as fallback.")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.feature_extractor import FEATURE_NAMES, FEATURE_GROUPS

# ── CONFIG ────────────────────────────────────────────────────────────────────
DATASET_CSV  = "data/dataset.csv"
MODEL_DIR    = "model"
RANDOM_SEED  = 42
TEST_SIZE    = 0.20
# ─────────────────────────────────────────────────────────────────────────────


def load_and_validate(csv_path):
    if not os.path.exists(csv_path):
        print(f"ERROR: {csv_path} not found.")
        print("  Run:  python utils/prepare_dataset.py  first.")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    print(f"  Rows      : {len(df):,}")
    print(f"  Features  : {len(FEATURE_NAMES)}")
    print(f"  Labels    :\n{df['label'].value_counts().to_string()}")

    missing = [f for f in FEATURE_NAMES if f not in df.columns]
    if missing:
        print(f"  ERROR: Missing feature columns: {missing}")
        sys.exit(1)

    # Drop rows with any NaN in feature columns
    before = len(df)
    df = df.dropna(subset=FEATURE_NAMES)
    if len(df) < before:
        print(f"  Dropped {before - len(df)} rows with NaN features.")

    X = df[FEATURE_NAMES].values.astype(np.float32)
    y = df['label'].values
    return X, y


def build_models(n_classes):
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_split=4,
        min_samples_leaf=2,
        max_features='sqrt',
        class_weight='balanced',
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    models = {"RandomForest": rf}

    if HAS_XGB:
        models["XGBoost"] = XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.08,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            use_label_encoder=False,
            eval_metric='mlogloss',
            random_state=RANDOM_SEED,
        )
    else:
        models["GradientBoosting"] = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.08,
            subsample=0.8,
            random_state=RANDOM_SEED,
        )
    return models


def train_and_evaluate(model, name, X_tr, X_te, y_tr, y_te, le):
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    acc = accuracy_score(y_te, y_pred)
    f1  = f1_score(y_te, y_pred, average='weighted')

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    cv_sc = cross_val_score(model, X_tr, y_tr, cv=cv, scoring='accuracy', n_jobs=-1)

    print(f"\n  ── {name} ──")
    print(f"     Test accuracy : {acc:.4f}")
    print(f"     F1 (weighted) : {f1:.4f}")
    print(f"     CV accuracy   : {cv_sc.mean():.4f} ± {cv_sc.std():.4f}")
    print(f"\n  {classification_report(y_te, y_pred, target_names=le.classes_)}")
    return acc, f1, y_pred


def plot_confusion_matrix(y_te, y_pred, le, path):
    cm = confusion_matrix(y_te, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=le.classes_, yticklabels=le.classes_, ax=ax,
                linewidths=0.5)
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('Actual', fontsize=12)
    ax.set_title('Confusion Matrix — Cognitive Load Classifier', fontsize=13)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def plot_feature_importance(model, name, path):
    if not hasattr(model, 'feature_importances_'):
        return
    imp = model.feature_importances_
    idx = np.argsort(imp)

    fig, ax = plt.subplots(figsize=(9, 7))
    colors = []
    group_color_map = {
        "Pressure":   "#3266ad",
        "Slant":      "#854F0B",
        "Spacing":    "#0F6E56",
        "Form/Shape": "#533AB7",
    }
    feat_to_group = {}
    for grp, feats in FEATURE_GROUPS.items():
        for f in feats:
            feat_to_group[f] = grp

    bar_colors = [group_color_map.get(feat_to_group.get(FEATURE_NAMES[i], ''), '#888') for i in idx]
    ax.barh([FEATURE_NAMES[i].replace('_', ' ') for i in idx],
            imp[idx], color=bar_colors, alpha=0.85)
    ax.set_xlabel('Feature Importance', fontsize=11)
    ax.set_title(f'Graphological Feature Importance — {name}', fontsize=12)

    # Legend
    from matplotlib.patches import Patch
    legend = [Patch(color=c, label=g) for g, c in group_color_map.items()]
    ax.legend(handles=legend, loc='lower right', fontsize=9)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def plot_group_importance(model, path):
    if not hasattr(model, 'feature_importances_'):
        return
    imp = model.feature_importances_
    name_to_imp = dict(zip(FEATURE_NAMES, imp))
    group_imp = {g: sum(name_to_imp.get(f, 0) for f in feats)
                 for g, feats in FEATURE_GROUPS.items()}

    fig, ax = plt.subplots(figsize=(6, 4))
    colors = ["#3266ad", "#854F0B", "#0F6E56", "#533AB7"]
    bars = ax.bar(group_imp.keys(), group_imp.values(), color=colors, alpha=0.85, width=0.5)
    for bar, val in zip(bars, group_imp.values()):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{val:.3f}', ha='center', va='bottom', fontsize=10)
    ax.set_ylabel('Total Importance', fontsize=11)
    ax.set_title('Feature Group Importance', fontsize=12)
    ax.set_ylim(0, max(group_imp.values()) * 1.2)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def train():
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("\n[1/5] Loading dataset...")
    X, y_raw = load_and_validate(DATASET_CSV)

    print("\n[2/5] Encoding labels & splitting...")
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    print(f"  Classes (encoded): {list(zip(le.classes_, range(len(le.classes_))))}")

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )
    print(f"  Train: {len(X_tr):,}  |  Test: {len(X_te):,}")

    print("\n[3/5] Scaling features (StandardScaler)...")
    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr)
    X_te_sc = scaler.transform(X_te)

    print("\n[4/5] Training models...")
    models = build_models(len(le.classes_))
    best_model, best_name, best_acc, best_pred, best_f1 = None, None, 0, None, 0

    for name, model in models.items():
        acc, f1, y_pred = train_and_evaluate(
            model, name, X_tr_sc, X_te_sc, y_tr, y_te, le
        )
        if acc > best_acc:
            best_acc, best_f1 = acc, f1
            best_model, best_name, best_pred = model, name, y_pred

    print(f"\n  Best: {best_name}  (accuracy={best_acc:.4f})")

    print("\n[5/5] Saving artifacts...")
    joblib.dump(best_model, os.path.join(MODEL_DIR, "cognitive_load_model.pkl"))
    joblib.dump(scaler,     os.path.join(MODEL_DIR, "scaler.pkl"))
    joblib.dump(le,         os.path.join(MODEL_DIR, "label_encoder.pkl"))

    info = {
        "model_name":    best_name,
        "accuracy":      round(best_acc, 4),
        "f1_score":      round(best_f1, 4),
        "classes":       list(le.classes_),
        "feature_names": FEATURE_NAMES,
        "feature_groups": FEATURE_GROUPS,
        "n_train":       len(X_tr),
        "n_test":        len(X_te),
    }
    with open(os.path.join(MODEL_DIR, "model_info.json"), "w") as f:
        json.dump(info, f, indent=2)
    print(f"  Saved: model/model_info.json")

    plot_confusion_matrix(y_te, best_pred, le,
                          os.path.join(MODEL_DIR, "confusion_matrix.png"))
    plot_feature_importance(best_model, best_name,
                            os.path.join(MODEL_DIR, "feature_importance.png"))
    plot_group_importance(best_model,
                          os.path.join(MODEL_DIR, "feature_group_importance.png"))

    print(f"\n✅ Training complete!")
    print(f"   Model : {MODEL_DIR}/cognitive_load_model.pkl")
    print(f"   Acc   : {best_acc:.4f}  |  F1: {best_f1:.4f}")


if __name__ == "__main__":
    train()
