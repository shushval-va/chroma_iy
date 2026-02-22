"""Chroma + PuLID: Photorealistic image generation with identity preservation.
RunPod Serverless Handler.

Architecture:
  - Base model: Chroma (Apache 2.0, uncensored, Flux-schnell derivative)
  - Identity: PuLID cross-attention face embedding injection
  - Quantization: 8-bit (optimum-quanto) for 24GB VRAM
  - All models are public, downloaded at Docker build time.
"""

import base64
import gc
import io
import os
import sys
import types

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from safetensors.torch import load_file
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import normalize, resize

import runpod

# --- Path configuration ---
MODEL_DIR = "/app/models"
CHROMA_DIR = os.path.join(MODEL_DIR, "Chroma")
PULID_CODE_DIR = "/app/PuLID"
sys.path.insert(0, PULID_CODE_DIR)

# PuLID architecture constants (Chroma: 19 double + 38 single blocks)
DOUBLE_INTERVAL = 2
SINGLE_INTERVAL = 4
NUM_DOUBLE = 19
NUM_SINGLE = 38


# ---------------------------------------------------------------------------
# PuLID model wrapper
# ---------------------------------------------------------------------------
class PuLIDModel(nn.Module):
    """Loads PuLID encoder (IDFormer) and cross-attention (PerceiverAttentionCA)
    modules. Architecture-agnostic: works with both Flux and Chroma."""

    def __init__(self):
        super().__init__()
        from pulid.encoders_transformer import IDFormer, PerceiverAttentionCA

        self.pulid_encoder = IDFormer()

        num_ca = NUM_DOUBLE // DOUBLE_INTERVAL + NUM_SINGLE // SINGLE_INTERVAL
        if NUM_DOUBLE % DOUBLE_INTERVAL != 0:
            num_ca += 1
        if NUM_SINGLE % SINGLE_INTERVAL != 0:
            num_ca += 1
        self.pulid_ca = nn.ModuleList(
            [PerceiverAttentionCA() for _ in range(num_ca)]
        )

    def load_pretrained(self, path):
        state_dict = load_file(path)
        grouped = {}
        for k, v in state_dict.items():
            module = k.split(".")[0]
            grouped.setdefault(module, {})
            grouped[module][k[len(module) + 1 :]] = v
        for module_name, sd in grouped.items():
            print(f"  loading PuLID module: {module_name}")
            getattr(self, module_name).load_state_dict(sd, strict=True)


# ---------------------------------------------------------------------------
# Monkey-patched ChromaTransformer2DModel.forward with PuLID injection
# ---------------------------------------------------------------------------
def chroma_forward_with_pulid(
    self,
    hidden_states,
    encoder_hidden_states=None,
    timestep=None,
    img_ids=None,
    txt_ids=None,
    attention_mask=None,
    joint_attention_kwargs=None,
    controlnet_block_samples=None,
    controlnet_single_block_samples=None,
    return_dict=True,
    controlnet_blocks_repeat=False,
):
    """Drop-in replacement for ChromaTransformer2DModel.forward that injects
    PuLID cross-attention between transformer blocks.

    PuLID data is read from ``self._pulid_data`` dict:
      - ``ca``: nn.ModuleList of PerceiverAttentionCA modules
      - ``embedding``: id embedding tensor  (B, N, D)
      - ``weight``: scalar influence weight
    When ``_pulid_data`` is ``None``, falls back to the original forward.
    """
    from diffusers.models.modeling_outputs import Transformer2DModelOutput

    pulid = getattr(self, "_pulid_data", None)
    if pulid is None:
        return self._original_forward(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timestep,
            img_ids=img_ids,
            txt_ids=txt_ids,
            attention_mask=attention_mask,
            joint_attention_kwargs=joint_attention_kwargs,
            controlnet_block_samples=controlnet_block_samples,
            controlnet_single_block_samples=controlnet_single_block_samples,
            return_dict=return_dict,
            controlnet_blocks_repeat=controlnet_blocks_repeat,
        )

    pulid_ca = pulid["ca"]
    pulid_emb = pulid["embedding"]
    pulid_weight = pulid["weight"]

    # --- Original embedding logic (mirrors diffusers source) ---
    hidden_states = self.x_embedder(hidden_states)
    timestep = timestep.to(hidden_states.dtype) * 1000
    input_vec = self.time_text_embed(timestep)
    pooled_temb = self.distilled_guidance_layer(input_vec)
    encoder_hidden_states = self.context_embedder(encoder_hidden_states)

    if txt_ids.ndim == 3:
        txt_ids = txt_ids[0]
    if img_ids.ndim == 3:
        img_ids = img_ids[0]

    ids = torch.cat((txt_ids, img_ids), dim=0)
    image_rotary_emb = self.pos_embed(ids)

    if joint_attention_kwargs is not None and "ip_adapter_image_embeds" in joint_attention_kwargs:
        ip_adapter_image_embeds = joint_attention_kwargs.pop("ip_adapter_image_embeds")
        ip_hidden_states = self.encoder_hid_proj(ip_adapter_image_embeds)
        joint_attention_kwargs.update({"ip_hidden_states": ip_hidden_states})

    num_single = len(self.single_transformer_blocks)
    num_double = len(self.transformer_blocks)
    ca_idx = 0

    # --- Double (joint) blocks ---
    for i, block in enumerate(self.transformer_blocks):
        img_offset = 3 * num_single
        txt_offset = img_offset + 6 * num_double
        img_mod = img_offset + 6 * i
        text_mod = txt_offset + 6 * i
        temb = torch.cat(
            (pooled_temb[:, img_mod : img_mod + 6], pooled_temb[:, text_mod : text_mod + 6]),
            dim=1,
        )

        encoder_hidden_states, hidden_states = block(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            temb=temb,
            image_rotary_emb=image_rotary_emb,
            attention_mask=attention_mask,
            joint_attention_kwargs=joint_attention_kwargs,
        )

        if controlnet_block_samples is not None:
            interval_control = int(np.ceil(num_double / len(controlnet_block_samples)))
            if controlnet_blocks_repeat:
                hidden_states = hidden_states + controlnet_block_samples[i % len(controlnet_block_samples)]
            else:
                hidden_states = hidden_states + controlnet_block_samples[i // interval_control]

        # PuLID injection after every DOUBLE_INTERVAL-th block
        if i % DOUBLE_INTERVAL == 0:
            ca_mod = pulid_ca[ca_idx]
            ca_dtype = next(ca_mod.parameters()).dtype
            _emb = pulid_emb.to(device=hidden_states.device, dtype=ca_dtype)
            _img = hidden_states.to(ca_dtype)
            hidden_states = hidden_states + pulid_weight * ca_mod(_emb, _img).to(hidden_states.dtype)
            ca_idx += 1

    # --- Merge for single blocks ---
    hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
    txt_len = encoder_hidden_states.shape[1]

    # --- Single blocks ---
    for i, block in enumerate(self.single_transformer_blocks):
        start_idx = 3 * i
        temb = pooled_temb[:, start_idx : start_idx + 3]

        hidden_states = block(
            hidden_states=hidden_states,
            temb=temb,
            image_rotary_emb=image_rotary_emb,
            attention_mask=attention_mask,
            joint_attention_kwargs=joint_attention_kwargs,
        )

        if controlnet_single_block_samples is not None:
            interval_control = int(np.ceil(num_single / len(controlnet_single_block_samples)))
            hidden_states[:, txt_len:, ...] = (
                hidden_states[:, txt_len:, ...] + controlnet_single_block_samples[i // interval_control]
            )

        # PuLID injection on image part only
        if i % SINGLE_INTERVAL == 0:
            ca_mod = pulid_ca[ca_idx]
            ca_dtype = next(ca_mod.parameters()).dtype
            _emb = pulid_emb.to(device=hidden_states.device, dtype=ca_dtype)
            img_part = hidden_states[:, txt_len:, ...]
            _img = img_part.to(ca_dtype)
            img_part = img_part + pulid_weight * ca_mod(_emb, _img).to(img_part.dtype)
            hidden_states = torch.cat([hidden_states[:, :txt_len, ...], img_part], dim=1)
            ca_idx += 1

    # --- Final projection ---
    hidden_states = hidden_states[:, txt_len:, ...]
    temb = pooled_temb[:, -2:]
    hidden_states = self.norm_out(hidden_states, temb)
    output = self.proj_out(hidden_states)

    if not return_dict:
        return (output,)
    return Transformer2DModelOutput(sample=output)


# ---------------------------------------------------------------------------
# Face embedding extraction
# ---------------------------------------------------------------------------
def get_id_embedding(image_np, app, clip_vision, face_helper, eva_mean, eva_std, device, dtype):
    """Extract combined InsightFace + EVA-CLIP face embedding for PuLID.

    Args:
        image_np: RGB numpy array [0, 255].
    Returns:
        id_embedding tensor (B, num_queries, output_dim).
    """
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    # InsightFace embedding
    face_info = app.get(image_bgr)
    if len(face_info) > 0:
        face_info = sorted(
            face_info,
            key=lambda x: (x["bbox"][2] - x["bbox"][0]) * (x["bbox"][3] - x["bbox"][1]),
        )[-1]
        id_ante = torch.from_numpy(face_info["embedding"]).unsqueeze(0).to(device, dtype)
    else:
        id_ante = None

    # facexlib align
    face_helper.clean_all()
    face_helper.read_image(image_bgr)
    face_helper.get_face_landmarks_5(only_center_face=True)
    face_helper.align_warp_face()
    if len(face_helper.cropped_faces) == 0:
        raise RuntimeError("Face alignment failed — no face detected")
    align_face = face_helper.cropped_faces[0]

    # fallback InsightFace on aligned face
    if id_ante is None:
        from insightface.model_zoo import get_model as get_insightface_model
        handler_ante = get_insightface_model(
            os.path.join(MODEL_DIR, "insightface", "models", "antelopev2", "glintr100.onnx"),
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        handler_ante.prepare(ctx_id=0)
        id_ante = torch.from_numpy(handler_ante.get_feat(align_face)).unsqueeze(0).to(device, dtype)

    # Parse face mask
    from pulid.utils import img2tensor
    inp = img2tensor(align_face, bgr2rgb=True).unsqueeze(0).to(device) / 255.0
    parsing_out = face_helper.face_parse(normalize(inp, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))[0]
    parsing_out = parsing_out.argmax(dim=1, keepdim=True)
    bg_labels = [0, 16, 18, 7, 8, 9, 14, 15]
    bg = sum(parsing_out == lbl for lbl in bg_labels).bool()
    white = torch.ones_like(inp)
    gray = 0.299 * inp[:, 0:1] + 0.587 * inp[:, 1:2] + 0.114 * inp[:, 2:3]
    gray = gray.repeat(1, 3, 1, 1)
    face_features = torch.where(bg, white, gray)

    # EVA-CLIP transform + forward
    face_features = resize(face_features, clip_vision.image_size, InterpolationMode.BICUBIC)
    face_features = normalize(face_features, eva_mean, eva_std).to(device, dtype)
    id_cond_vit, id_vit_hidden = clip_vision(face_features, return_all_features=False, return_hidden=True, shuffle=False)
    id_cond_vit = id_cond_vit / id_cond_vit.norm(2, 1, True)

    id_cond = torch.cat([id_ante, id_cond_vit], dim=-1)
    return id_cond, id_vit_hidden


# ---------------------------------------------------------------------------
# Global model state
# ---------------------------------------------------------------------------
pipe = None
pulid_model = None
face_app = None
clip_vision = None
face_helper_global = None
eva_mean = None
eva_std = None


def load_models():
    global pipe, pulid_model, face_app, clip_vision, face_helper_global, eva_mean, eva_std

    device = "cuda"
    dtype = torch.bfloat16

    # --- 1. Chroma pipeline with 8-bit quantization ---
    print("[1/5] Loading Chroma pipeline...")
    from diffusers import ChromaPipeline
    from transformers import T5EncoderModel
    from optimum.quanto import freeze, qint8, quantize

    text_encoder = T5EncoderModel.from_pretrained(CHROMA_DIR, subfolder="text_encoder", torch_dtype=dtype)
    quantize(text_encoder, weights=qint8)
    freeze(text_encoder)

    pipe = ChromaPipeline.from_pretrained(CHROMA_DIR, text_encoder=text_encoder, torch_dtype=dtype)
    quantize(pipe.transformer, weights=qint8)
    freeze(pipe.transformer)
    pipe.to(device)
    print("  Chroma loaded (transformer + T5 quantized to int8)")

    # --- 2. PuLID model ---
    print("[2/5] Loading PuLID model...")
    pulid_model = PuLIDModel()
    pulid_path = os.path.join(MODEL_DIR, "pulid_flux_v0.9.1.safetensors")
    pulid_model.load_pretrained(pulid_path)
    pulid_model.to(device, dtype)
    pulid_model.eval()
    print("  PuLID loaded")

    # Monkey-patch transformer forward (once)
    pipe.transformer._original_forward = pipe.transformer.forward
    pipe.transformer._pulid_data = None
    pipe.transformer.forward = types.MethodType(chroma_forward_with_pulid, pipe.transformer)
    print("  Transformer forward patched for PuLID injection")

    # --- 3. InsightFace ---
    print("[3/5] Loading InsightFace (antelopev2)...")
    from insightface.app import FaceAnalysis
    face_app = FaceAnalysis(
        name="antelopev2",
        root=os.path.join(MODEL_DIR, "insightface"),
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    face_app.prepare(ctx_id=0, det_size=(640, 640))
    print("  InsightFace loaded")

    # --- 4. EVA-CLIP ---
    print("[4/5] Loading EVA-CLIP...")
    from eva_clip import create_model_and_transforms
    from eva_clip.constants import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD
    model_full, _, _ = create_model_and_transforms("EVA02-CLIP-L-14-336", "eva_clip", force_custom_clip=True)
    clip_vision = model_full.visual.to(device, dtype)
    clip_vision.eval()
    _mean = getattr(clip_vision, "image_mean", OPENAI_DATASET_MEAN)
    _std = getattr(clip_vision, "image_std", OPENAI_DATASET_STD)
    eva_mean = tuple(_mean) if not isinstance(_mean, (list, tuple)) else tuple(_mean)
    eva_std = tuple(_std) if not isinstance(_std, (list, tuple)) else tuple(_std)
    del model_full
    print("  EVA-CLIP loaded")

    # --- 5. facexlib ---
    print("[5/5] Loading facexlib...")
    from facexlib.parsing import init_parsing_model
    from facexlib.utils.face_restoration_helper import FaceRestoreHelper
    face_helper_global = FaceRestoreHelper(
        upscale_factor=1, face_size=512, crop_ratio=(1, 1),
        det_model="retinaface_resnet50", save_ext="png", device=device,
    )
    face_helper_global.face_parse = init_parsing_model(model_name="bisenet", device=device)
    print("  facexlib loaded")

    gc.collect()
    torch.cuda.empty_cache()
    print("=== All models loaded ===")


# ---------------------------------------------------------------------------
# RunPod handler
# ---------------------------------------------------------------------------
def handler(job):
    inp = job["input"]

    # Decode face image
    face_b64 = inp["face_image"]
    face_bytes = base64.b64decode(face_b64)
    face_pil = Image.open(io.BytesIO(face_bytes)).convert("RGB")
    face_np = np.array(face_pil)

    prompt = inp.get("prompt", "a woman, portrait photo")
    width = inp.get("width", 768)
    height = inp.get("height", 1024)
    num_steps = inp.get("num_steps", 26)
    guidance_scale = inp.get("guidance_scale", 0.0)
    id_weight = inp.get("id_weight", 1.0)
    seed = inp.get("seed", 0)
    if seed == 0:
        seed = int(torch.seed() & 0xFFFFFFFF)

    device = "cuda"
    dtype = torch.bfloat16

    # Extract face embedding
    print(f"Extracting face embedding...")
    id_cond, id_vit_hidden = get_id_embedding(
        face_np, face_app, clip_vision, face_helper_global, eva_mean, eva_std, device, dtype,
    )
    with torch.no_grad():
        id_embedding = pulid_model.pulid_encoder(id_cond, id_vit_hidden)
    print(f"  id_embedding shape: {id_embedding.shape}")

    # Activate PuLID on transformer
    pipe.transformer._pulid_data = {
        "ca": pulid_model.pulid_ca,
        "embedding": id_embedding,
        "weight": id_weight,
    }

    # Generate
    print(f"Generating {width}x{height}, steps={num_steps}, seed={seed}, id_weight={id_weight}")
    generator = torch.Generator(device=device).manual_seed(seed)
    result = pipe(
        prompt=prompt,
        width=width,
        height=height,
        num_inference_steps=num_steps,
        guidance_scale=guidance_scale,
        generator=generator,
    )
    image = result.images[0]

    # Deactivate PuLID
    pipe.transformer._pulid_data = None

    # Encode output
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    image_b64 = base64.b64encode(buf.getvalue()).decode()

    gc.collect()
    torch.cuda.empty_cache()

    return {"image": image_b64, "seed": seed}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    load_models()
    runpod.serverless.start({"handler": handler})
