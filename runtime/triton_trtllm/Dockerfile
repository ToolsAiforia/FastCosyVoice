FROM ghcr.io/astral-sh/uv:latest AS uv-builder

FROM nvcr.io/nvidia/tritonserver:25.06-trtllm-python-py3

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
  git+https://github.com/ToolsAiforia/FastCosyVoice@aeb3a11fa614fb3b7e1299e38aed08c00f3286d6

RUN pip uninstall onnxruntime onnxruntime-gpu -y && \
    uv pip install --system --break-system-packages --no-cache-dir onnxruntime-gpu>=1.17 

WORKDIR /workspace


ADD ./model_repo_cosyvoice2 /model_repo
ADD ./CosyVoice2-0.5B /workspace/CosyVoice2-0.5B
ADD ./cosyvoice2_llm /workspace/cosyvoice2_llm
ADD ./trt_engines_bfloat16 /workspace/trt_engines_bfloat16
