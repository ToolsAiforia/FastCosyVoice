FROM ghcr.io/astral-sh/uv:latest AS uv-builder

FROM nvcr.io/nvidia/tritonserver:26.01-trtllm-python-py3

# UV
COPY --from=uv-builder /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_SYSTEM_PYTHON=1

RUN uv pip install \
  --system --break-system-packages --no-cache-dir --no-build-isolation \
  setuptools wheel pip cython 

RUN --mount=type=ssh uv pip install \
  --system --break-system-packages --no-cache-dir --no-build-isolation \
  git+https://github.com/ToolsAiforia/FastCosyVoice@47c2ec7fc8fceecd3aa4e7221230cf878bb6b7f6

# RUN pip uninstall onnxruntime onnxruntime-gpu -y && \
#     uv pip install --system --break-system-packages --no-cache-dir onnxruntime-gpu>=1.17

WORKDIR /workspace
