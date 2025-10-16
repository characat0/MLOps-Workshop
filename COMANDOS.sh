cd "/home/jhampits/IA/IA GEN UNI/20251012"
python3 -m venv venvuni
source venvuni/bin/activate


## # PARA EL NMINTS
pip install torch torchvision torchaudio matplotlib pillow tqdm typer

## PARA IR PROBANDO

docker build .
## Para crear el docker 
docker build . -t mnist_clasification_jts

## Para ejecutar comando dentro del docker
docker run -it mnist_clasification_jts bash
docker run -it mnist_clasification_jts /bin/bash

### Para correr el RUN por default q enesta en DOCKERFILE
# EJM : ENTRYPOINT ["conda", "run", "-n", "uni_deep_learning", "python", "src/inference.py"]

docker run uni_deep_learning "https://raw.githubusercontent.com/lABrass/mnist-png/master/train/0/0006d579-9bff-42f4-b631-1a1b0f71a64f.png"