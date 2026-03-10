ARG BASE_IMAGE=nvcr.io/nvidia/tritonserver:24.12-py3
FROM ${BASE_IMAGE}

USER root

COPY scripts/triton/requirements-gtv2-runtime.txt /tmp/requirements-gtv2-runtime.txt

RUN python3 -m pip install --no-cache-dir --break-system-packages -r /tmp/requirements-gtv2-runtime.txt \
    && python3 - <<'PY'
import numpy
import pandas
import pyarrow
import torch

print("numpy", numpy.__version__)
print("pandas", pandas.__version__)
print("pyarrow", pyarrow.__version__)
print("torch", torch.__version__)
PY

WORKDIR /opt/tritonserver
