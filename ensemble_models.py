"""Ансамбли и полносвязные нейронные сети."""

import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import (
    RandomForestClassifier,
    HistGradientBoostingClassifier,
    StackingClassifier,
)
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

from grader_contracts.ensemble_models import EnsembleInput, EnsembleResult


def _prepare_features(features):
    if isinstance(features, pd.DataFrame):
        features = pd.get_dummies(features, drop_first=True)
        return features.values.astype(float)
    return np.array(features, dtype=float)


def run_ensemble_experiment(data: EnsembleInput) -> EnsembleResult:
    X_train = _prepare_features(data.train_features)
    y_train = np.array(data.train_target, dtype=float)
    X_test = _prepare_features(data.test_features)
    y_test = np.array(data.test_target, dtype=float)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model_name = "Stacking_Ensemble_SVC_HGB_RF"
    model = StackingClassifier(
        estimators=[
            ('svc', SVC(C=1.0, gamma='scale', kernel='rbf', probability=True, random_state=123)),
            ('hgb', HistGradientBoostingClassifier(
                max_iter=500, max_depth=10, learning_rate=0.05,
                l2_regularization=0.1, min_samples_leaf=20,
                random_state=123, early_stopping=True
            )),
            ('rf', RandomForestClassifier(
                n_estimators=1000, max_depth=15, min_samples_split=2,
                random_state=123
            ))
        ],
        final_estimator=LogisticRegression(C=1.0, max_iter=1000),
        cv=3
    )
    model.fit(X_train_scaled, y_train)

    probabilities = model.predict_proba(X_test_scaled)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    roc_auc = float(roc_auc_score(y_test, probabilities))
    accuracy = float(accuracy_score(y_test, predictions))
    precision = float(precision_score(y_test, predictions, zero_division=0))
    recall = float(recall_score(y_test, predictions, zero_division=0))
    f1 = float(f1_score(y_test, predictions, zero_division=0))

    metrics = {
        "roc_auc": roc_auc,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

    try:
        try:
            with open("experiment_results.json", "r", encoding="utf-8") as f:
                results = json.load(f)
        except FileNotFoundError:
            results = {}
        results["final_model"] = metrics
        with open("experiment_results.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
    except Exception:
        pass

    return EnsembleResult(
        model_name=model_name,
        probabilities=probabilities,
        predictions=predictions,
        roc_auc=roc_auc,
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
    )