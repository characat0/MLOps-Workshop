FROM continuumio/miniconda3:latest
WORKDIR /app
SHELL ["/bin/bash", "-lc"]

# 1) Crear env base (sin torch/vision)
COPY environment.yml .
RUN conda env create -f environment.yml

# 2) Instalar torch/vision CPU en el MISMO origen y versión compatible
#    Nota: el índice CPU oficial evita wheels CUDA.
RUN conda run -n uni_deep_learning python -m pip install --no-cache-dir \
    torch==2.4.1 torchvision==0.19.1 \
    --index-url https://download.pytorch.org/whl/cpu

# 3) (Opcional pero recomendado) sanity check de versiones y del op 'nms'
RUN conda run -n uni_deep_learning python - <<'PY'
import torch, torchvision
from torchvision.ops import nms
print("OK torch", torch.__version__, "vision", torchvision.__version__)
PY

# 4) Estructura del proyecto y training en build
RUN mkdir -p src data/models
COPY src/train.py src/train.py
#RUN conda run -n uni_deep_learning python src/train.py

# 5) Inference
COPY src/inference.py src/inference.py
ENTRYPOINT ["conda", "run", "-n", "uni_deep_learning", "python"]
CMD ["src/inference.py"]