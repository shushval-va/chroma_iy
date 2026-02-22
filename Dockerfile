FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV HF_HUB_DOWNLOAD_TIMEOUT=600
ENV HF_HUB_ENABLE_HF_TRANSFER=1

# Accept HF token as build arg for faster downloads (optional)
ARG HF_TOKEN=""
ENV HF_TOKEN=${HF_TOKEN}

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

# 1. Chroma model (diffusers format, Apache 2.0)
# Download sequentially (max_workers=1) to avoid OOM during build
RUN python -c "\
from huggingface_hub import snapshot_download; \
import os; \
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '0'; \
print('=== Downloading Chroma model (sequential) ==='); \
snapshot_download('lodestones/Chroma', local_dir='/app/models/Chroma', max_workers=1); \
print('Chroma download complete'); \
"

# 2. PuLID model weights
RUN python -c "\
from huggingface_hub import hf_hub_download; \
print('=== Downloading PuLID model ==='); \
hf_hub_download('guozinan/PuLID', 'pulid_flux_v0.9.1.safetensors', local_dir='/app/models'); \
print('PuLID download complete'); \
"

# 3. InsightFace antelopev2 (public mirror)
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
