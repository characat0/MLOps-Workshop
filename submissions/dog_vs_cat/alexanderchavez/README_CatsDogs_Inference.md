# 🐶🐱 Cats vs Dogs – Inference (Docker)

Contenedor ligero para **inferencia de imágenes** (*Cats vs Dogs*) entrenado con **TensorFlow/Keras**.  
El modelo determina si una imagen corresponde a un **perro (`dog`)** o un **gato (`cat`)**.

---

## 🚀 Requisitos

- Tener **Docker Desktop** (Windows/Mac) o **Docker Engine** (Linux) en ejecución.
- El modelo debe existir en:  
  `models/cats_dogs_simple.keras`

---

## 🏗️ Build de la imagen

Ejecuta desde la carpeta del proyecto  
`submissions/dog_vs_cat/alexanderchavez`:

```bash
docker build -t catsdogs-infer .
```

---

## 🧠 Uso básico

### 🔹 Inferencia con una **URL externa**

```bash
docker run --rm catsdogs-infer "https://example.com/mi_perro.jpg"
```

Salida esperada:
```
dog
```

Modo verboso (muestra probabilidades y umbral usado):
```bash
docker run --rm catsdogs-infer "https://example.com/mi_gato.jpg" --verbose
```

Salida ejemplo:
```
cat
p_dog=0.0321  p_cat=0.9679  threshold=0.5
```

---

### 🔹 Inferencia con una **imagen local**

**Linux / macOS:**
```bash
docker run --rm -v "$PWD:/work" catsdogs-infer "/work/data/val_sorted/cats/cat.10003.jpg"
```

**Windows PowerShell:**
```powershell
docker run --rm -v "$((Get-Location).Path):/work" catsdogs-infer "/work/data/val_sorted/dogs/dog.0.jpg"
```

---

## ⚙️ Dockerfile usado

```dockerfile
FROM tensorflow/tensorflow:2.16.1

WORKDIR /app

RUN pip install --no-cache-dir pillow requests

COPY models/cats_dogs_simple.keras models/cats_dogs_simple.keras
COPY src/inference.py src/inference.py

ENTRYPOINT ["python", "src/inference.py"]
```

---

## 🧩 Archivos requeridos

```
alexanderchavez/
├── Dockerfile
├── README.md
├── models/
│   └── cats_dogs_simple.keras
└── src/
    └── inference.py
```

---

## 💡 Notas adicionales

- Imagen base: `tensorflow/tensorflow:2.16.1` (CPU)
- Para GPU: usa `tensorflow/tensorflow:2.16.1-gpu` y ejecuta con `--gpus all`
- Dependencias incluidas:  
  `tensorflow`, `pillow`, `requests`
- Salida simple: `cat` o `dog`
- Salida extendida: con `--verbose`, imprime probabilidades

---

**Autor:** Alexander Chávez  
**Proyecto:** MLOps Workshop – Cats vs Dogs 🧠🐾  
**Versión:** 1.0.0  
**Fecha:** Octubre 2025
