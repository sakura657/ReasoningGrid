FROM runpod/pytorch:2.2.1-py3.10-cuda12.1.1-devel-ubuntu22.04

# ---- 用户/权限 ----
ARG NB_USER=appuser
ARG NB_UID=1000
ARG NB_GID=1000

USER root
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    git build-essential ninja-build python3-dev && \
    rm -rf /var/lib/apt/lists/*

RUN groupadd -g ${NB_GID} ${NB_USER} && \
    useradd -m -s /bin/bash -u ${NB_UID} -g ${NB_GID} ${NB_USER} && \
    mkdir -p /app /workspace && chown -R ${NB_UID}:${NB_GID} /app /workspace

# ---- Python & 依赖 ----
RUN python3 -m pip install --upgrade pip

# 与 CUDA 12.1 匹配的 PyTorch
RUN pip3 install --no-cache-dir torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu121

# 其余核心依赖（注意：此处不要带续行反斜杠到下一条 RUN）
RUN pip3 install --no-cache-dir \
    vllm==0.7.2 \
    "datasets>=2.17.0,<3" \
    setuptools \
    lighteval==0.8.1 \
    math-verify==0.5.2

# flash-attn 单独安装（可能会编译）
RUN pip3 install --no-cache-dir --no-build-isolation flash-attn

# fsspec/s3fs 固定到顶层导出 url_to_fs 的版本（2024.3.1+，取 2024.6.1）
RUN pip3 install --no-cache-dir "fsspec==2024.6.1" "s3fs==2024.6.1"

# 构建期自检：确认可顶层导入 url_to_fs
RUN python3 - <<'PY'
import fsspec; print("fsspec version:", fsspec.__version__)
from fsspec import url_to_fs
print("url_to_fs import: OK")
PY

# ---- 运行用户 & 工作目录 ----
USER ${NB_UID}:${NB_GID}
ENV HOME=/home/${NB_USER}
WORKDIR /workspace

CMD ["/bin/bash"]