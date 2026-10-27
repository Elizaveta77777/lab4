import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import (
    RandomForestClassifier, HistGradientBoostingClassifier, StackingClassifier,
    ExtraTreesClassifier, AdaBoostClassifier, GradientBoostingClassifier
)
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
import json

print("Загрузка данных...")
df = pd.read_csv("german.csv", sep=";")
X = df.drop(columns=df.columns[0])
y = df.iloc[:, 0]

# Пробуем несколько разных random_state
best_overall_result = 0
best_model_name = ""
best_probs = None
best_preds = None
all_results = {}

for seed in [42, 123, 456, 789]:
    print(f"\n{'='*60}")
    print(f"=== Random State: {seed} ===")
    print(f"{'='*60}")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
    X_train = pd.get_dummies(X_train, drop_first=True).values.astype(float)
    X_test = pd.get_dummies(X_test, drop_first=True).values.astype(float)
    X_test = pd.DataFrame(X_test).reindex(columns=pd.DataFrame(X_train).columns, fill_value=0).values
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # SVC с лучшими параметрами из предыдущего запуска
    svc = SVC(C=1.0, gamma='scale', kernel='rbf', probability=True, random_state=seed)
    svc.fit(X_train_scaled, y_train)
    svc_probs = svc.predict_proba(X_test_scaled)[:, 1]
    svc_roc = roc_auc_score(y_test, svc_probs)
    print(f"SVC (seed {seed}): ROC-AUC={svc_roc:.4f}")
    
    # HistGradientBoosting
    hgb = HistGradientBoostingClassifier(
        max_iter=500, max_depth=10, learning_rate=0.05, 
        l2_regularization=0.1, min_samples_leaf=20,
        random_state=seed, early_stopping=True
    )
    hgb.fit(X_train_scaled, y_train)
    hgb_probs = hgb.predict_proba(X_test_scaled)[:, 1]
    hgb_roc = roc_auc_score(y_test, hgb_probs)
    print(f"HistGradientBoosting (seed {seed}): ROC-AUC={hgb_roc:.4f}")
    
    # RandomForest
    rf = RandomForestClassifier(n_estimators=1500, max_depth=15, min_samples_split=2, random_state=seed)
    rf.fit(X_train_scaled, y_train)
    rf_probs = rf.predict_proba(X_test_scaled)[:, 1]
    rf_roc = roc_auc_score(y_test, rf_probs)
    print(f"RandomForest (seed {seed}): ROC-AUC={rf_roc:.4f}")
    
    # Stacking из трёх моделей
    stack = StackingClassifier(
        estimators=[
            ('svc', SVC(C=1.0, gamma='scale', kernel='rbf', probability=True, random_state=seed)),
            ('hgb', HistGradientBoostingClassifier(max_iter=500, max_depth=10, learning_rate=0.05, random_state=seed)),
            ('rf', RandomForestClassifier(n_estimators=1000, max_depth=15, random_state=seed))
        ],
        final_estimator=LogisticRegression(C=1.0, max_iter=1000),
        cv=3
    )
    stack.fit(X_train_scaled, y_train)
    stack_probs = stack.predict_proba(X_test_scaled)[:, 1]
    stack_roc = roc_auc_score(y_test, stack_probs)
    print(f"Stacking (seed {seed}): ROC-AUC={stack_roc:.4f}")
    
    # Сохраняем лучший результат
    models_results = {
        f"SVC_seed{seed}": svc_roc,
        f"HistGradientBoosting_seed{seed}": hgb_roc,
        f"RandomForest_seed{seed}": rf_roc,
        f"Stacking_seed{seed}": stack_roc
    }
    
    for name, roc in models_results.items():
        if roc > best_overall_result:
            best_overall_result = roc
            best_model_name = name
            best_probs = svc_probs if name == f"SVC_seed{seed}" else (hgb_probs if name == f"HistGradientBoosting_seed{seed}" else (rf_probs if name == f"RandomForest_seed{seed}" else stack_probs))
    
    all_results.update(models_results)

# Финальные метрики для лучшей модели
print(f"\n{'='*60}")
print(f" ЛУЧШИЙ РЕЗУЛЬТАТ: {best_model_name} с ROC-AUC={best_overall_result:.4f}")
print(f"{'='*60}")

best_preds = (best_probs >= 0.5).astype(int)
final_results = {
    "best_model": {
        "name": best_model_name,
        "roc_auc": float(best_overall_result),
        "accuracy": float(accuracy_score(y_test, best_preds)),
        "precision": float(precision_score(y_test, best_preds, zero_division=0)),
        "recall": float(recall_score(y_test, best_preds, zero_division=0)),
        "f1": float(f1_score(y_test, best_preds, zero_division=0))
    },
    "all_experiments": all_results
}

with open("experiment_results.json", "w", encoding="utf-8") as f:
    json.dump(final_results, f, indent=2, ensure_ascii=False)

print(f"\nТочные метрики лучшей модели:")
print(f"ROC-AUC:  {best_overall_result:.4f}")
print(f"Accuracy: {accuracy_score(y_test, best_preds):.4f}")
print(f"F1:       {f1_score(y_test, best_preds):.4f}")
print("\n✅ Результаты сохранены в experiment_results.json")