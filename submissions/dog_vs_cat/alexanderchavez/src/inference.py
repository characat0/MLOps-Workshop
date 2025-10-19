# src/inference.py
import argparse
from pathlib import Path
from io import BytesIO
import sys

import requests
from PIL import Image
from tensorflow import keras

IMG_SIZE = (224, 224)
THRESH = 0.5

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "cats_dogs_simple.keras"

def eprint(*a):
    print(*a, file=sys.stderr, flush=True)

# ---------- Compat para deserializar capas de augment ----------
class RandomFlipCompat(keras.layers.Layer):
    """Envoltura compatible que ignora 'data_format' al reconstruir."""
    def __init__(self, mode="horizontal", seed=None, **kwargs):
        super().__init__(**kwargs)
        # delegamos en la capa oficial de tu versión actual
        self._inner = keras.layers.RandomFlip(mode=mode, seed=seed)

    def call(self, x, training=None):
        return self._inner(x, training=training)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"mode": getattr(self._inner, "mode", "horizontal")})
        return cfg

    @classmethod
    def from_config(cls, config):
        # ¡clave! algunos modelos guardan 'data_format' y aquí no existe
        config = dict(config)  # copia
        config.pop("data_format", None)
        return cls(**config)

class RandomRotationCompat(keras.layers.Layer):
    """Envoltura compatible que ignora 'data_format' y mantiene firma estable."""
    def __init__(self, factor=0.0, fill_mode="reflect", fill_value=0.0,
                 interpolation="bilinear", seed=None, **kwargs):
        super().__init__(**kwargs)
        self._inner = keras.layers.RandomRotation(
            factor=factor,
            fill_mode=fill_mode,
            fill_value=fill_value,
            interpolation=interpolation,
            seed=seed,
        )

    def call(self, x, training=None):
        return self._inner(x, training=training)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"factor": getattr(self._inner, "factor", 0.0)})
        return cfg

    @classmethod
    def from_config(cls, config):
        config = dict(config)
        config.pop("data_format", None)
        return cls(**config)

# Registrar TODOS los alias que suelen venir en el .keras
CUSTOM_OBJECTS = {
    # nombres simples
    "RandomFlip": RandomFlipCompat,
    "RandomRotation": RandomRotationCompat,
    # alias totalmente calificados vistos en distintos dumps
    "keras.layers.preprocessing.random_flip.RandomFlip": RandomFlipCompat,
    "keras.layers.preprocessing.random_rotation.RandomRotation": RandomRotationCompat,
    "keras.src.layers.preprocessing.random_flip.RandomFlip": RandomFlipCompat,
    "keras.src.layers.preprocessing.random_rotation.RandomRotation": RandomRotationCompat,
    # a veces vienen como image_preprocessing.*
    "keras.layers.preprocessing.image_preprocessing.RandomFlip": RandomFlipCompat,
    "keras.layers.preprocessing.image_preprocessing.RandomRotation": RandomRotationCompat,
}

def load_model_or_exit():
    if not MODEL_PATH.exists():
        eprint(f"[ERROR] Modelo no encontrado en: {MODEL_PATH}")
        sys.exit(2)
    try:
        return keras.models.load_model(
            MODEL_PATH,
            compile=False,
            safe_mode=False,             # permite objetos no puros
            custom_objects=CUSTOM_OBJECTS
        )
    except Exception as ex:
        eprint("[ERROR] Falló al cargar el modelo:", ex)
        sys.exit(2)

def load_image(source: str) -> Image.Image:
    try:
        if source.lower().startswith(("http://", "https://")):
            r = requests.get(source, timeout=30)
            r.raise_for_status()
            return Image.open(BytesIO(r.content)).convert("RGB")
        p = Path(source)
        if not p.exists():
            raise FileNotFoundError(f"No existe la imagen local: {p}")
        return Image.open(p).convert("RGB")
    except Exception as ex:
        eprint("[ERROR] No se pudo leer la imagen:", ex)
        sys.exit(3)

def predict_label(model, img: Image.Image, threshold: float = THRESH):
    img = img.resize(IMG_SIZE)
    x = keras.utils.img_to_array(img)[None, ...]
    try:
        p_dog = float(model.predict(x, verbose=0)[0, 0])
    except Exception as ex:
        eprint("[ERROR] Falló la predicción:", ex)
        sys.exit(4)
    label = "dog" if p_dog > threshold else "cat"
    return label, p_dog

def main():
    ap = argparse.ArgumentParser(description="Cats vs Dogs inference (Keras)")
    ap.add_argument("source", help="Ruta local o URL de la imagen")
    ap.add_argument("--threshold", type=float, default=THRESH)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    model = load_model_or_exit()
    img = load_image(args.source)
    label, p_dog = predict_label(model, img, args.threshold)

    print(label, flush=True)
    if args.verbose:
        print(f"p_dog={p_dog:.4f}  p_cat={1.0-p_dog:.4f}  threshold={args.threshold}", flush=True)

if __name__ == "__main__":
    main()
