FROM python:3.11

RUN pip install scikit-learn pandas joblib cloudpickle

WORKDIR /app

COPY data/heart.csv data/heart.csv

COPY src src

RUN mkdir -p data/model

RUN python src/train.py

ENTRYPOINT [ "python", "src/docs.py" ]