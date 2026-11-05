"""Entraînement et comparaison de trois modèles de classification du churn."""
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "raw" / "churn.csv"
MODELS_DIR = ROOT / "models"
MODEL_VERSION = "1.0.0"

FEATURES = [
    "age",
    "anciennete",
    "revenu",
    "nb_reclamations",
    "montant_mensuel",
    "retards_paiement",
]
TARGET = "churn"

df = pd.read_csv(DATA_FILE)

X = df[FEATURES]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

models = {
    "logistic_regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000)),
    ]),
    "decision_tree": DecisionTreeClassifier(max_depth=5, random_state=42),
    "random_forest": RandomForestClassifier(n_estimators=200, random_state=42),
}

results = {}

for name, clf in models.items():
    print(f"\nEntraînement : {name}")
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    results[name] = {
        "accuracy": float(round(accuracy_score(y_test, predictions), 4)),
        "precision": float(round(precision_score(y_test, predictions, zero_division=0), 4)),
        "recall": float(round(recall_score(y_test, predictions, zero_division=0), 4)),
        "f1": float(round(f1_score(y_test, predictions, zero_division=0), 4)),
    }
    print(results[name])

# Sélection du Random Forest pour la suite du TP
selected = "random_forest"
model = models[selected]

MODELS_DIR.mkdir(parents=True, exist_ok=True)
joblib.dump(model, MODELS_DIR / "model.pkl", compress=3)

# Métadonnées : version, métriques, variables -> utilisées par /model/info
metadata = {
    "model_name": "Random Forest",
    "model_version": MODEL_VERSION,
    "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    "features": FEATURES,
    "train_size": len(X_train),
    "test_size": len(X_test),
    "metrics": results,
    "selected": selected,
}
(MODELS_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

print("\nModèle sauvegardé dans models/model.pkl")
print("Métadonnées sauvegardées dans models/metadata.json")
