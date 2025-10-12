FROM continuumio/miniconda3:latest

WORKDIR /app


COPY environment.yml .


RUN conda env create -f environment.yml


RUN mkdir -p src
RUN mkdir -p data/models


RUN conda run -n uni_deep_learning python -c 'import torchvision; torchvision.datasets.MNIST(root="data", train=True, download=True, transform=torchvision.transforms.ToTensor())'

COPY src/train.py src/train.py
RUN conda run -n uni_deep_learning python src/train.py


