FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV HF_HUB_DOWNLOAD_TIMEOUT=600
ENV HF_HUB_ENABLE_HF_TRANSFER=1

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget g++ libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python dependencies
RUN pip install --no-cache-dir \
    runpod==1.7.9 \
    brotlicffi \
    "diffusers>=0.34.0" \
    transformers \
    accelerate \
    safetensors \
    sentencepiece \
    protobuf \
    insightface \
    "onnxruntime-gpu==1.20.0" \
    opencv-python-headless \
    huggingface_hub \
    hf_transfer \
    einops \
    timm \
    ftfy \
    facexlib \
    "optimum-quanto>=0.2.7" \
    peft

# Clone PuLID repo (encoder code + EVA-CLIP)
RUN git clone --depth 1 https://github.com/ToTheBeginning/PuLID.git /app/PuLID \
    && rm -rf /app/PuLID/.git /app/PuLID/docs /app/PuLID/example_inputs

# Verify core imports
RUN python -c "\
import torch; print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}'); \
import diffusers; print(f'Diffusers {diffusers.__version__}'); \
from diffusers import ChromaPipeline; print('ChromaPipeline OK'); \
import runpod; print('RunPod OK'); \
import insightface; print('InsightFace OK'); \
import facexlib; print('facexlib OK'); \
"

# ===== Download ALL models at build time (all public, no auth needed) =====
# Using wget instead of Python to stream files directly to disk (zero RAM overhead)

ENV HF_BASE=https://huggingface.co

# 1. Chroma model — ONLY diffusers-format files (16 files, ~28GB)
#    The repo has 100+ files (~1.3TB) including old checkpoints we don't need
RUN mkdir -p /app/models/Chroma/scheduler /app/models/Chroma/text_encoder \
    /app/models/Chroma/tokenizer /app/models/Chroma/transformer /app/models/Chroma/vae \
    && echo "=== Downloading Chroma pipeline configs ===" \
    && wget -q -O /app/models/Chroma/model_index.json \
       "$HF_BASE/lodestones/Chroma/resolve/main/model_index.json" \
    && wget -q -O /app/models/Chroma/scheduler/scheduler_config.json \
       "$HF_BASE/lodestones/Chroma/resolve/main/scheduler/scheduler_config.json" \
    && wget -q -O /app/models/Chroma/tokenizer/added_tokens.json \
       "$HF_BASE/lodestones/Chroma/resolve/main/tokenizer/added_tokens.json" \
    && wget -q -O /app/models/Chroma/tokenizer/special_tokens_map.json \
       "$HF_BASE/lodestones/Chroma/resolve/main/tokenizer/special_tokens_map.json" \
    && wget -q -O /app/models/Chroma/tokenizer/spiece.model \
       "$HF_BASE/lodestones/Chroma/resolve/main/tokenizer/spiece.model" \
    && wget -q -O /app/models/Chroma/tokenizer/tokenizer_config.json \
       "$HF_BASE/lodestones/Chroma/resolve/main/tokenizer/tokenizer_config.json" \
    && wget -q -O /app/models/Chroma/text_encoder/config.json \
       "$HF_BASE/lodestones/Chroma/resolve/main/text_encoder/config.json" \
    && wget -q -O /app/models/Chroma/text_encoder/model.safetensors.index.json \
       "$HF_BASE/lodestones/Chroma/resolve/main/text_encoder/model.safetensors.index.json" \
    && wget -q -O /app/models/Chroma/transformer/config.json \
       "$HF_BASE/lodestones/Chroma/resolve/main/transformer/config.json" \
    && wget -q -O /app/models/Chroma/transformer/diffusion_pytorch_model.safetensors.index.json \
       "$HF_BASE/lodestones/Chroma/resolve/main/transformer/diffusion_pytorch_model.safetensors.index.json" \
    && wget -q -O /app/models/Chroma/vae/config.json \
       "$HF_BASE/lodestones/Chroma/resolve/main/vae/config.json" \
    && echo "Chroma configs downloaded"

# 1b. Chroma large files (each downloaded separately for Docker layer caching)
RUN echo "=== Downloading Chroma text_encoder shard 1/2 ===" \
    && wget -nv -O /app/models/Chroma/text_encoder/model-00001-of-00002.safetensors \
       "$HF_BASE/lodestones/Chroma/resolve/main/text_encoder/model-00001-of-00002.safetensors"

RUN echo "=== Downloading Chroma text_encoder shard 2/2 ===" \
    && wget -nv -O /app/models/Chroma/text_encoder/model-00002-of-00002.safetensors \
       "$HF_BASE/lodestones/Chroma/resolve/main/text_encoder/model-00002-of-00002.safetensors"

RUN echo "=== Downloading Chroma transformer shard 1/2 ===" \
    && wget -nv -O /app/models/Chroma/transformer/diffusion_pytorch_model-00001-of-00002.safetensors \
       "$HF_BASE/lodestones/Chroma/resolve/main/transformer/diffusion_pytorch_model-00001-of-00002.safetensors"

RUN echo "=== Downloading Chroma transformer shard 2/2 ===" \
    && wget -nv -O /app/models/Chroma/transformer/diffusion_pytorch_model-00002-of-00002.safetensors \
       "$HF_BASE/lodestones/Chroma/resolve/main/transformer/diffusion_pytorch_model-00002-of-00002.safetensors"

RUN echo "=== Downloading Chroma VAE ===" \
    && wget -nv -O /app/models/Chroma/vae/diffusion_pytorch_model.safetensors \
       "$HF_BASE/lodestones/Chroma/resolve/main/vae/diffusion_pytorch_model.safetensors"

# 2. PuLID model weights
RUN echo "=== Downloading PuLID model ===" \
    && mkdir -p /app/models \
    && wget -nv -O /app/models/pulid_flux_v0.9.1.safetensors \
       "$HF_BASE/guozinan/PuLID/resolve/main/pulid_flux_v0.9.1.safetensors"

# 3. InsightFace antelopev2 (public mirror, small files)
RUN python -c "\
from huggingface_hub import snapshot_download; \
print('=== Downloading antelopev2 ==='); \
snapshot_download('DIAMONIK7777/antelopev2', local_dir='/app/models/insightface/models/antelopev2'); \
print('antelopev2 download complete'); \
"

# 4. EVA-CLIP model (auto-downloaded by create_model_and_transforms)
RUN python -c "\
import sys; sys.path.insert(0, '/app/PuLID'); \
from eva_clip import create_model_and_transforms; \
print('=== Downloading EVA-CLIP ==='); \
model, _, _ = create_model_and_transforms('EVA02-CLIP-L-14-336', 'eva_clip', force_custom_clip=True); \
print('EVA-CLIP download complete'); \
"

# 5. facexlib models (retinaface + bisenet)
RUN python -c "\
from facexlib.utils.face_restoration_helper import FaceRestoreHelper; \
from facexlib.parsing import init_parsing_model; \
print('=== Downloading facexlib models ==='); \
helper = FaceRestoreHelper(upscale_factor=1, face_size=512, crop_ratio=(1,1), \
    det_model='retinaface_resnet50', save_ext='png', device='cpu'); \
parser = init_parsing_model(model_name='bisenet', device='cpu'); \
print('facexlib download complete'); \
"

# Disable network access for runtime (all models are local)
ENV HF_HUB_OFFLINE=1

# Copy handler
COPY handler.py /app/handler.py

CMD ["python", "handler.py"]
