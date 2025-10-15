# src/train.py
import os
import json
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score

from preprocess import preprocessing_min  # función compartida

# ---------- Rutas ----------
CURRENT_FILE = __file__
DATA_FOLDER  = os.path.join(os.path.dirname(CURRENT_FILE), "..", "data")
DATA_FILE    = os.path.join(DATA_FOLDER, "train.txt")      # "text;label"
MODEL_DIR    = os.path.join(DATA_FOLDER, "model")
MODEL_PATH   = os.path.join(MODEL_DIR, "model.joblib")
REPORT_PATH  = os.path.join(MODEL_DIR, "metrics.json")
CLASSES_PATH = os.path.join(MODEL_DIR, "classes.json")

os.makedirs(MODEL_DIR, exist_ok=True)

def main():
    # ---------- Carga de datos ----------
    df = pd.read_csv(DATA_FILE, sep=";", header=None, names=["text", "label"], encoding="utf-8")

    # ---------- Split ----------
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
    )

    # ---------- Pipeline: TF-IDF -> RandomForest ----------
    pipeline = Pipeline(steps=[
        ("tfidf", TfidfVectorizer(
            preprocessor=preprocessing_min,   # usa la función compartida
            tokenizer=None,
            ngram_range=(1, 2),
            max_features=5000
        )),
        ("rf", RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            random_state=42,
            n_jobs=-1
        ))
    ])

    # ---------- Entrenamiento ----------
    pipeline.fit(X_train, y_train)

    # ---------- Evaluación ----------
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1w = f1_score(y_test, y_pred, average="weighted")
    report = classification_report(y_test, y_pred, output_dict=True)

    print("Accuracy:", acc)
    print("F1 (weighted):", f1w)
    print("\nClassification report:")
    print(classification_report(y_test, y_pred))

    # ---------- Guardar métricas ----------
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump({"accuracy": acc, "f1_weighted": f1w, "report": report}, f, ensure_ascii=False, indent=2)

    # ---------- Guardar modelo ----------
    joblib.dump(pipeline, MODEL_PATH)

    # ---------- Guardar clases ----------
    classes = pipeline.named_steps["rf"].classes_.tolist()
    with open(CLASSES_PATH, "w", encoding="utf-8") as f:
        json.dump({"classes": classes}, f, ensure_ascii=False, indent=2)

    print(f"\nModelo guardado en:   {MODEL_PATH}")
    print(f"Métricas guardadas en:{REPORT_PATH}")
    print(f"Clases guardadas en:  {CLASSES_PATH}")

if __name__ == "__main__":
    main()
