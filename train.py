import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from skimage.feature import hog, local_binary_pattern

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             classification_report, confusion_matrix, log_loss)

import mlflow
import mlflow.sklearn
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

import skops.io as sio
from joblib import Parallel, delayed

def get_features(img_path):
    img = cv2.imread(img_path)
    if len(img.shape) > 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (128, 128))
    
    hog_feats = hog(img, orientations=9, pixels_per_cell=(16, 16),
                    cells_per_block=(2, 2), block_norm='L2-Hys')
    
    P = 8
    R = 1
    lbp = local_binary_pattern(img, P, R, method="uniform")
    
    n_bins = int(lbp.max() + 1)
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
    
    return np.hstack([hog_feats, lbp_hist])

def build_feature_matrix_and_labels(paths, labels):
    X = Parallel(n_jobs=-1, backend="multiprocessing")(
        delayed(get_features)(p) for p in paths
    )
    return np.array(X), np.array(labels)

def load_paths_and_labels(base_path, classes):
    paths, labels = [], []
    for name in classes:
        folder = os.path.join(base_path, name)
        files = os.listdir(folder)
        for f in files:
            paths.append(os.path.join(folder, f))
            labels.append(name)
    return paths, labels

if __name__ == '__main__':
    
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("multiclass_classification_experiments_2")

    train_base_path = "NEU-DET/train/images"
    val_base_path = "NEU-DET/validation/images"
    class_names = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]

    train_paths, train_labels = load_paths_and_labels(train_base_path, class_names)
    val_paths, val_labels = load_paths_and_labels(val_base_path, class_names)

    print("Preparing dataset and features from Train and Validation sets...")
    x_train, y_train_raw = build_feature_matrix_and_labels(train_paths, train_labels)
    x_val, y_val_raw = build_feature_matrix_and_labels(val_paths, val_labels)

    le = LabelEncoder()
    y_train = le.fit_transform(y_train_raw)
    y_val = le.transform(y_val_raw)

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_val_scaled = scaler.transform(x_val)

    models = {
        'SVC_linear': SVC(kernel='linear', probability=True, random_state=42),
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
        'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
        'XGBoost': XGBClassifier(eval_metric='mlogloss', random_state=42),
        'LightGBM': LGBMClassifier(random_state=42, verbosity=-1),
        'CatBoost': CatBoostClassifier(iterations=200, verbose=0, random_state=42)
    }

    model_leaderboard = []
    best_score = -1.0
    best_model = None
    best_name = None
    best_run_id = None

    for name, clf in models.items():
        print(f"Training {name}...")
        with mlflow.start_run(run_name=name) as run:
            mlflow.log_params(clf.get_params())
            clf.fit(x_train_scaled, y_train)

            preds = clf.predict(x_val_scaled)
            probs = clf.predict_proba(x_val_scaled) if hasattr(clf, "predict_proba") else None

            acc = accuracy_score(y_val, preds)
            
            # Compute Average Metrics (Macro and Weighted configurations)
            p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(y_val, preds, average='macro')
            p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(y_val, preds, average='weighted')
            
            mlflow.log_metric('validation_accuracy', float(acc))
            mlflow.log_metric('val_precision_macro', float(p_macro))
            mlflow.log_metric('val_recall_macro', float(r_macro))
            mlflow.log_metric('val_f1_macro', float(f1_macro))
            mlflow.log_metric('val_precision_weighted', float(p_weighted))
            mlflow.log_metric('val_recall_weighted', float(r_weighted))
            mlflow.log_metric('val_f1_weighted', float(f1_weighted))
            
            if probs is not None:
                mlflow.log_metric('val_log_loss', float(log_loss(y_val, probs)))

            # Per-Class Metric Collection
            p_per_class, r_per_class, f1_per_class, _ = precision_recall_fscore_support(y_val, preds, labels=range(len(class_names)))
            for idx, cls in enumerate(le.classes_):
                mlflow.log_metric(f'val_precision_{cls}', float(p_per_class[idx]))
                mlflow.log_metric(f'val_recall_{cls}', float(r_per_class[idx]))
                mlflow.log_metric(f'val_f1_{cls}', float(f1_per_class[idx]))

            report = classification_report(y_val, preds, target_names=le.classes_)
            mlflow.log_text(report, f'classification_report_{name}.txt')

            cm = confusion_matrix(y_val, preds)
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', xticklabels=le.classes_, yticklabels=le.classes_, cmap='Blues', ax=ax)
            ax.set_ylabel('True Class')
            ax.set_xlabel('Predicted Class')
            ax.set_title(f'Confusion Matrix - {name}')
            plt.tight_layout()
            
            mlflow.log_figure(fig, f'confusion_matrix_{name}.png')
            plt.close(fig)

            # Store stats for the sorting leaderboard logic loop
            model_leaderboard.append({
                'name': name,
                'accuracy': acc,
                'f1_macro': f1_macro,
                'f1_weighted': f1_weighted,
                'score_key': (f1_macro + acc) / 2.0  # Composite balancing standard for ranking sorting
            })

            if f1_macro > best_score:
                best_score = f1_macro
                best_model = clf
                best_name = name
                best_run_id = run.info.run_id

    # Rank and display leaderboard structures from best to worst performance evaluation benchmarks
    model_leaderboard = sorted(model_leaderboard, key=lambda x: x['score_key'], reverse=True)
    
    print("\n" + "="*50)
    print("FINAL EVALUATION LEADERBOARD (Ranked Best to Worst)")
    print("="*50)
    for rank, entry in enumerate(model_leaderboard, 1):
        print(f"Rank {rank}: {entry['name']}")
        print(f"  -> Accuracy:     {entry['accuracy']:.4f}")
        print(f"  -> Macro F1:      {entry['f1_macro']:.4f}")
        print(f"  -> Weighted F1:   {entry['f1_weighted']:.4f}")
        print("-"*50)

    if best_model is not None:
        state_payload = {
            'best_model': best_model,
            'scaler': scaler,
            'encoder': le,
            'classes': class_names,
            'best_name': best_name,
            'best_run_id': best_run_id
        }
        sio.dump(state_payload, '.training_state.skops')
        print(f"\nSaved tracking runtime state. Winning model: '{best_name}'")