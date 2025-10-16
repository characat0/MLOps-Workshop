FROM continuumio/miniconda3:latest

WORKDIR /app


COPY environment.yml .


RUN conda env create -f environment.yml


RUN mkdir -p src
RUN mkdir -p data/models 


COPY src/train.py src/train.py
RUN conda run -n uni_deep_learning python src/train.py

COPY src/inference.py src/inference.py

ENTRYPOINT ["conda", "run", "-n", "uni_deep_learning", "python", "src/inference.py"]
