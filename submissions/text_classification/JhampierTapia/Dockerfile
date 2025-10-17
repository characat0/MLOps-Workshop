FROM python:3.11

COPY src src

RUN pip install -r src/requirements.txt

#RUN pip install scikit-learn pandas joblib cloudpickle

WORKDIR /app

COPY data/train.txt data/train.txt

RUN mkdir -p data/model

RUN python src/train.py

ENTRYPOINT [ "python", "src/docs.py" ]