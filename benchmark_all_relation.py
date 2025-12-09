import argparse
import os
import json
from src.modules.image_encoder import ImageEncoder
from src.pipeline import Pipeline
from src.utils import set_ms_adapter
import torch
from PIL import Image
from safetensors import safe_open
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
from transformers import (
    CLIPImageProcessor,
    CLIPTextConfig,
    CLIPModel,
    CLIPProcessor,
)
from transformers import AutoImageProcessor, AutoModel
import torch.nn.functional as F

# ==========================
# Load LoRA Weights
# ==========================
def load_lora_weights(ckpt_path):
    state_dict = {}
    with safe_open(ckpt_path, framework="pt", device=0) as f:
        for k in f.keys():
            state_dict[k] = f.get_tensor(k)
    return state_dict


# ==========================
# Metrics: Helper Functions
# ==========================

def cosine_sim_1d(a, b):
    """Cosine similarity for two 1D feature vectors."""
    a = a.float()
    b = b.float()
    return F.cosine_similarity(a, b, dim=0).item()


def compute_clip_image_similarity(clip_model, clip_processor, img1, img2, device):
    """
    Computes CLIP-I: similarity between generated and identity image.
    """
    batch = clip_processor(images=[img1, img2], return_tensors="pt", padding=True)
    batch = {k: v.to(device) for k, v in batch.items()}

    with torch.no_grad():
        feats = clip_model.get_image_features(batch["pixel_values"])
        feats = feats / feats.norm(dim=-1, keepdim=True)

    return cosine_sim_1d(feats[0], feats[1])


def compute_clip_text_similarity(clip_model, clip_processor, img, text, device):
    """
    Computes CLIP-T or CLIP-R: similarity between image and text prompt.
    """
    batch = clip_processor(text=[text], images=[img], return_tensors="pt", padding=True)
    batch = {k: v.to(device) for k, v in batch.items()}

    with torch.no_grad():
        img_feat = clip_model.get_image_features(batch["pixel_values"])
        txt_feat = clip_model.get_text_features(batch["input_ids"])

        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
        txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)

    return cosine_sim_1d(img_feat[0], txt_feat[0])


def compute_dino_similarity(dino_model, dino_processor, img1, img2, device):
    """
    Computes DINO similarity between generated and identity image.
    """
    batch = dino_processor(images=[img1, img2], return_tensors="pt")
    batch = {k: v.to(device) for k, v in batch.items()}

    with torch.no_grad():
        outputs = dino_model(**batch)
        feats = outputs.last_hidden_state.mean(dim=1)
        feats = feats / feats.norm(dim=-1, keepdim=True)

    return cosine_sim_1d(feats[0], feats[1])


def extract_relation_word(prompt):
    """
    Extracts relationship keyword for CLIP-R.
    Minimal rule-based extractor.
    """
    COMMON_RELATIONS = [
        "hugging", "holding", "kissing", "riding", "carrying", "touching",
        "feeding", "playing", "shaking hands", "shaking", "standing on",
        "sitting on", "behind", "in front of", "fighting"
    ]

    low = prompt.lower()

    # Exact phrase match first (handles multiword)
    for r in COMMON_RELATIONS:
        if r in low:
            return r

    # Fallback: choose verb-ish token
    tokens = low.split()
    for t in tokens:
        if t.endswith(("ing", "ed")):
            return t

    # Final fallback
    return tokens[0]


# ==========================
# Main Script
# ==========================
def main():
    parser = argparse.ArgumentParser(description="Inference Script")

    # Model paths
    parser.add_argument("--pretrained_model_name_or_path", type=str,
                        default="stabilityai/stable-diffusion-xl-base-1.0")
    parser.add_argument("--clip_model_name_or_path", type=str,
                        default="laion/CLIP-ViT-bigG-14-laion2B-39B-b160k")
    parser.add_argument("--ms_ckpt", type=str,
                        default="./checkpoints/MS-Diffusion/ms_adapter.bin")
    parser.add_argument("--clipself_ckpt", type=str,
                        default="./checkpoints/local_image_encoder/epoch_6.pt")

    # Generation settings
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--scale", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_inference_steps", type=int, default=30)

    # LoRA
    parser.add_argument("--load_lora", action="store_true")
    parser.add_argument("--no_load_lora", dest="load_lora", action="store_false")
    parser.set_defaults(load_lora=True)

    # Output
    parser.add_argument("--output_dir", type=str, default="results/all/")
    parser.add_argument("--negative_prompt", type=str,
                        default="monochrome, lowres, bad anatomy, worst quality, low quality")

    # Device
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    # Load prompts JSON
    with open("Benchmark/prompts.json", "r") as f:
        prompt_entries = json.load(f)

    # Load MS diffusion adapter
    ms_state_dict = torch.load(args.ms_ckpt)
    image_encoder_state_dict = {}
    for key, value in ms_state_dict.items():
        if key == "image_proj":
            for k, v in value.items():
                image_encoder_state_dict["resampler." + k] = v
        elif key == "dummy_image_tokens":
            image_encoder_state_dict[key] = value
        else:
            ms_adapter_state_dict = value

    # Initialize UNet
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="unet"
    ).to(args.device, torch.float16)

    adapter_modules = set_ms_adapter(unet, scale=args.scale)
    adapter_modules.load_state_dict(ms_adapter_state_dict)

    # Text encoder config
    text_encoder_config = CLIPTextConfig.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder"
    )

    # Image encoder
    image_encoder = ImageEncoder(
        args.clip_model_name_or_path,
        clipself_pretrained=args.clipself_ckpt,
        dim=1280,
        depth=4,
        dim_head=64,
        heads=20,
        num_queries=16,
        output_dim=unet.config.cross_attention_dim,
        ff_mult=4,
        latent_init_mode="grounding",
        phrase_embeddings_dim=text_encoder_config.projection_dim,
    ).to(args.device, dtype=torch.float16)

    image_encoder.load_state_dict(image_encoder_state_dict, strict=False)


    os.makedirs(args.output_dir, exist_ok=True)

    # ==========================
    # Load CLIP & DINO models
    # ==========================
    clip_model_name = "openai/clip-vit-large-patch14"
    clip_model = CLIPModel.from_pretrained(clip_model_name).to(args.device)
    clip_processor = CLIPProcessor.from_pretrained(clip_model_name)

    dino_model_name = "facebook/dino-vits16"
    dino_model = AutoModel.from_pretrained(dino_model_name).to(args.device)
    dino_processor = AutoImageProcessor.from_pretrained(dino_model_name)

    # ==========================
    # Global metric accumulators
    # ==========================
    total_clip_i = 0.0
    total_dino = 0.0
    total_clip_t = 0.0
    total_clip_r = 0.0
    count = 0

    # ==========================
    # Loop over prompts and images
    # ==========================
    i=0
    for entry in prompt_entries:
        i+=1
        sub1 = entry["sub1"]           # e.g., "benchmark/plushie_cat"
        prompt = entry["prompt"]       # text prompt
        phrase = entry["phrase"]       # e.g., "plushie_cat"
        model_name = entry.get("model", None)
         # Pipeline
        pipe = Pipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            unet=unet,
            image_encoder=image_encoder,
        ).to(device=args.device, dtype=torch.float16)

        # LoRA
        lora_weights = f"lora-weights/{model_name}"
        if args.load_lora:
            pipe.load_lora_weights(lora_weights)

        input_dir = os.path.join("RelationBench", sub1)
        if not os.path.isdir(input_dir):
            print(f"[WARN] Missing folder: {input_dir}")
            continue

        # mirror structure in output
        sub_output_dir = os.path.join(args.output_dir, f"{i}_{sub1}")
        os.makedirs(sub_output_dir, exist_ok=True)

        # iterate all .jpg files
        image_files = sorted(f for f in os.listdir(input_dir) if f.lower().endswith(".jpg"))
        if not image_files:
            print(f"[WARN] No .jpg files in {input_dir}")
            continue

        print(f"\n=== Evaluating {sub1} ({len(image_files)} images) ===")

        for fname in image_files:
            img_path = os.path.join(input_dir, fname)

            # Load input images (single identity image)
            images_path = [img_path]
            input_images = [Image.open(p).convert("RGB").resize((512, 512)) for p in images_path]

            # CLIP processors for MS-Diffusion concept images
            image_processor = CLIPImageProcessor()
            processed_images = [image_processor(images=input_images, return_tensors="pt").pixel_values]
            processed_images = torch.stack(processed_images, dim=0)

            image_processor_896 = CLIPImageProcessor(size=896, crop_size=896)
            processed_images_896 = [image_processor_896(images=input_images, return_tensors="pt").pixel_values]
            processed_images_896 = torch.stack(processed_images_896, dim=0)

            # ==========================
            # Run Inference
            # ==========================
            generator = torch.Generator(unet.device).manual_seed(args.seed)
            with torch.no_grad():
                result = pipe(
                    prompt=prompt,
                    negative_prompt=args.negative_prompt,
                    concept_images=processed_images,
                    concept_images_896=processed_images_896,
                    num_inference_steps=args.num_inference_steps,
                    boxes=[[[0, 0, 1, 1]]],
                    phrases=[[phrase]],
                    generator=generator,
                    num_images_per_prompt=args.num_samples
                )

            gen_image = result.images[0]
            out_path = os.path.join(sub_output_dir, fname)
            gen_image.save(out_path)
            print(f"Generated image saved to: {out_path}")

            # ==========================
            # Compute Metrics
            # ==========================
            identity_img = input_images[0]  # first concept image

            clip_i = compute_clip_image_similarity(
                clip_model, clip_processor, gen_image, identity_img, args.device
            )

            dino = compute_dino_similarity(
                dino_model, dino_processor, gen_image, identity_img, args.device
            )

            clip_t = compute_clip_text_similarity(
                clip_model, clip_processor, gen_image, prompt, args.device
            )

            relation_word = extract_relation_word(prompt)

            clip_r = compute_clip_text_similarity(
                clip_model, clip_processor, gen_image, relation_word, args.device
            )

            # accumulate
            total_clip_i += clip_i
            total_dino += dino
            total_clip_t += clip_t
            total_clip_r += clip_r
            count += 1

            print(f"{sub1}/{fname}: CLIP-I={clip_i:.3f}, DINO={dino:.3f}, "
                  f"CLIP-T={clip_t:.3f}, CLIP-R={clip_r:.3f}")

    # ==========================
    # Final average metrics
    # ==========================
    if count > 0:
        avg_metrics = {
            "CLIP-I": total_clip_i / count,
            "DINO": total_dino / count,
            "CLIP-T": total_clip_t / count,
            "CLIP-R": total_clip_r / count,
            "num_images": count,
        }
    else:
        avg_metrics = {
            "CLIP-I": 0.0,
            "DINO": 0.0,
            "CLIP-T": 0.0,
            "CLIP-R": 0.0,
            "num_images": 0,
        }

    print("\n=== FINAL AVERAGE METRICS ===")
    for k, v in avg_metrics.items():
        print(f"{k}: {v}")

    # Save averages to metrics.json (same filename as before)
    metrics_path = os.path.join(args.output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(avg_metrics, f, indent=4)

    print(f"\nAverage metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
