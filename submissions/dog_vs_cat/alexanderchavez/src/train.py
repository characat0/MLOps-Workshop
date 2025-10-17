from pathlib import Path
import zipfile, shutil, random, re
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

ROOT = Path.cwd().parent if Path.cwd().name == "notebook" else Path.cwd()
DATA = ROOT / "data"
MODELS = ROOT / "models"
MODELS.mkdir(parents=True, exist_ok=True)

TRAIN_ZIP = DATA / "train.zip"
TEST_ZIP  = DATA / "test1.zip" 


EXTRACT_TRAIN = DATA / "train"
if TRAIN_ZIP.exists():
    print(f"Descomprimiendo {TRAIN_ZIP.name}…")
    with zipfile.ZipFile(TRAIN_ZIP, "r") as zf:
        zf.extractall(EXTRACT_TRAIN)
    print("OK ->", EXTRACT_TRAIN)
else:
    print("⚠ No se encontró", TRAIN_ZIP)

if TEST_ZIP.exists():
    print(f"Descomprimiendo {TEST_ZIP.name}…")
    with zipfile.ZipFile(TEST_ZIP, "r") as zf:
        zf.extractall(DATA / "test")
    print("OK ->", DATA / "test")


candidates = [
    DATA / "train_train",
    DATA / "train" / "train",
    DATA / "train",
]
TRAIN_SRC = None
for c in candidates:
    if c.exists() and any(c.glob("*.jpg")):
        TRAIN_SRC = c
        break
assert TRAIN_SRC is not None, "No encuentro carpeta con .jpg (train_train / train/train / train)."
print("SRC_TRAIN:", TRAIN_SRC)


TRAIN_SORTED = DATA / "train_sorted"
CATS_DIR = TRAIN_SORTED / "cats"
DOGS_DIR = TRAIN_SORTED / "dogs"
for d in (CATS_DIR, DOGS_DIR): d.mkdir(parents=True, exist_ok=True)

pat_cat = re.compile(r'^cat\.\d+\.jpg$', re.IGNORECASE)
pat_dog = re.compile(r'^dog\.\d+\.jpg$', re.IGNORECASE)

moved_cats = moved_dogs = skipped = 0
for img in TRAIN_SRC.glob("*.jpg"):
    n = img.name
    if pat_cat.match(n):
        shutil.move(str(img), str(CATS_DIR / n)); moved_cats += 1
    elif pat_dog.match(n):
        shutil.move(str(img), str(DOGS_DIR / n)); moved_dogs += 1
    else:
        skipped += 1

print(f"Movidos -> cats={moved_cats}, dogs={moved_dogs}, ignorados={skipped}")
print("Destino:", TRAIN_SORTED)

VAL_SORTED = DATA / "val_sorted"
VAL_CATS, VAL_DOGS = VAL_SORTED / "cats", VAL_SORTED / "dogs"
for d in (VAL_CATS, VAL_DOGS): d.mkdir(parents=True, exist_ok=True)

def move_split(src, dst, frac=0.2, seed=1337):
    files = [p for p in src.glob("*.jpg")]
    random.Random(seed).shuffle(files)
    k = int(len(files) * frac)
    for p in files[:k]:
        shutil.move(str(p), str(dst / p.name))
    return k, len(files) - k

kc, rc = move_split(CATS_DIR, VAL_CATS)
kd, rd = move_split(DOGS_DIR, VAL_DOGS)
print(f"VAL movidas: cats={kc}, dogs={kd} · TRAIN remanente: cats={rc}, dogs={rd}")

IMG_SIZE = (224, 224)
BATCH = 32
SEED = 1337

_train = keras.utils.image_dataset_from_directory(
    TRAIN_SORTED, labels="inferred", label_mode="int",
    image_size=IMG_SIZE, batch_size=BATCH, seed=SEED, shuffle=True,
)
_val = keras.utils.image_dataset_from_directory(
    VAL_SORTED, labels="inferred", label_mode="int",
    image_size=IMG_SIZE, batch_size=BATCH, seed=SEED, shuffle=False,
)

class_names = tuple(_train.class_names)  # ('cats','dogs')
print("Clases:", class_names)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = _train.cache().prefetch(AUTOTUNE)
val_ds   = _val.cache().prefetch(AUTOTUNE)

from tensorflow import keras
from tensorflow.keras import layers

# --- Modelo sencillo con backbone preentrenado ---
base = keras.applications.MobileNetV2(
    input_shape=(*IMG_SIZE, 3), include_top=False, weights="imagenet"
)
base.trainable = False  # rápido y estable

data_aug = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.05),
], name="augment")

inp = keras.Input(shape=(*IMG_SIZE, 3))
x = data_aug(inp)
x = keras.applications.mobilenet_v2.preprocess_input(x)
x = base(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.2)(x)                    # un poco de regularización
out = layers.Dense(1, activation="sigmoid")(x)

model = keras.Model(inp, out)
model.compile(optimizer=keras.optimizers.Adam(1e-3),
              loss="binary_crossentropy",
              metrics=["accuracy"])

# --- Callbacks simples para parar a tiempo ---
early = keras.callbacks.EarlyStopping(
    monitor="val_accuracy", mode="max", patience=2, restore_best_weights=True
)

# Entrenamiento corto pero suficiente
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=5,             # 3 suele bastar; 5 da un pelín más de margen
    callbacks=[early],
    verbose=1
)


from pathlib import Path

# Asegura rutas
ROOT = Path.cwd().parent if Path.cwd().name == "notebook" else Path.cwd()
MODELS = ROOT / "models"
MODELS.mkdir(parents=True, exist_ok=True)  # 👈 crea la carpeta si no existe

MODEL_PATH = MODELS / "cats_dogs_simple.keras"
model.save(MODEL_PATH)
print("Modelo guardado en:", MODEL_PATH)