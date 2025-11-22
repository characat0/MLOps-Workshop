FROM continuumio/miniconda3:latest

WORKDIR /app


COPY environment.yml .


RUN conda env create -f environment.yml

RUN conda run -n uni_deep_learning pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cpu

RUN mkdir -p src
RUN mkdir -p data/models


COPY src/train.py src/train.py
RUN conda run -n uni_deep_learning python src/train.py

COPY src/inference.py src/inference.py

ENTRYPOINT ["conda", "run", "-n", "uni_deep_learning", "python", "src/inference.py"]