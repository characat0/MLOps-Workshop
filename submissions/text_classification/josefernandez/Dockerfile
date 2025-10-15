FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN pip install --no-cache-dir scikit-learn pandas numpy scipy joblib

WORKDIR /app

COPY src/ src/
COPY data/ data/

RUN mkdir -p data/model
RUN python src/train.py

ENTRYPOINT ["python", "src/predict.py"]
CMD ["i love this movie"]
