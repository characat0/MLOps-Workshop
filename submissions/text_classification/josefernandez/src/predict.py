# src/predict.py
import os
import json
import argparse
import joblib
import numpy as np

# Importar el módulo asegura que joblib pueda resolver referencias si fuese necesario
from preprocess import preprocessing_min  # aunque no lo usemos directo aquí

CURRENT_FILE = __file__
DATA_FOLDER  = os.path.join(os.path.dirname(CURRENT_FILE), "..", "data")
MODEL_DIR    = os.path.join(DATA_FOLDER, "model")
MODEL_PATH   = os.path.join(MODEL_DIR, "model.joblib")
CLASSES_PATH = os.path.join(MODEL_DIR, "classes.json")

def load_model_and_classes():
    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(f"No se encontró el modelo: {MODEL_PATH}. ¿Ya corriste train.py?")
    if not os.path.isfile(CLASSES_PATH):
        raise FileNotFoundError(f"No se encontró el archivo de clases: {CLASSES_PATH}. ¿Ya corriste train.py?")

    model = joblib.load(MODEL_PATH)
    with open(CLASSES_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)
    classes = meta["classes"]
    class2id = {c: i for i, c in enumerate(classes)}
    return model, classes, class2id

def main():
    parser = argparse.ArgumentParser(description="Predict emotion from text")
    parser.add_argument("text", type=str, help="Input text to classify")
    args = parser.parse_args()

    model, classes, class2id = load_model_and_classes()

    text = args.text
    pred_label = model.predict([text])[0]
    pred_id = class2id[pred_label]

    proba_vec = model.predict_proba([text])[0]
    pred_pct = float(proba_vec[pred_id] * 100.0)

    # Salida: "<id> - <label>: <pct>%"
    print(f"{pred_id} - {pred_label}: {pred_pct:.2f}%")

if __name__ == "__main__":
    main()
