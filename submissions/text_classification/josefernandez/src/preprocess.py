# src/preprocess.py
import re

def preprocessing_min(text, keep_digits=True):
    """
    Preprocesamiento básico:
      - lower
      - quitar HTML, URLs y emails
      - quitar puntuación (y opcionalmente dígitos)
      - colapsar espacios
    """
    text = "" if text is None else str(text)
    text = text.lower()
    # quitar HTML, URLs y emails
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"http[s]?://\S+|www\.\S+", " ", text)
    text = re.sub(r"\b[\w\.-]+@[\w\.-]+\.\w+\b", " ", text)
    # quitar puntuación (y opcionalmente dígitos)
    text = re.sub(r"[^a-z0-9\s]", " ", text) if keep_digits else re.sub(r"[^a-z\s]", " ", text)
    # colapsar espacios
    text = re.sub(r"\s+", " ", text).strip()
    return text
