FROM ghcr.io/astral-sh/uv:latest AS uv-builder

FROM nvcr.io/nvidia/tritonserver:25.11-trtllm-python-py3

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
  git+https://github.com/ToolsAiforia/FastCosyVoice@05160ced9bb5ef29958eee2b1ada42499bcec378

WORKDIR /workspace
