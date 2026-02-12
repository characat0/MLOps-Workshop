# Dockerfile (TensorFlow + pip)
FROM tensorflow/tensorflow:2.16.1

WORKDIR /app

RUN pip install --no-cache-dir pillow requests

COPY models/cats_dogs_simple.keras models/cats_dogs_simple.keras
COPY src/inference.py src/inference.py

ENTRYPOINT ["python", "src/inference.py"]
