"""CLIP multi-modal fusion example

Usage:
  python UniSRec/scripts/clip_fusion_example.py --meta meta.jsonl --text-field title --image-field image_path --out-dir /tmp/emb

This script computes CLIP text and image features, fuses them, projects to a fixed dim,
L2-normalizes and saves embeddings + builds a FAISS index.
"""
from pathlib import Path
import argparse
import json
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor
from PIL import Image

try:
    import faiss
except Exception:
    faiss = None


class FusionEmbedder(nn.Module):
    def __init__(self, clip_model_name: str = "openai/clip-vit-base-patch32", proj_dim: int = 256, device: str = "cpu"):
        super().__init__()
        self.device = device
        self.model = CLIPModel.from_pretrained(clip_model_name).to(device)
        self.processor = CLIPProcessor.from_pretrained(clip_model_name)

        # projection heads for text and image, then fusion
        clip_dim = self.model.config.projection_dim
        self.text_proj = nn.Sequential(nn.Linear(clip_dim, proj_dim), nn.ReLU())
        self.img_proj = nn.Sequential(nn.Linear(clip_dim, proj_dim), nn.ReLU())
        self.fusion = nn.Sequential(nn.Linear(proj_dim * 2, proj_dim), nn.ReLU())

    @torch.no_grad()
    def encode(self, texts: List[str], image_paths: Optional[List[Optional[str]]] = None, batch_size: int = 64):
        device = self.device
        text_feats = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            inputs = self.processor(text=batch, return_tensors="pt", padding=True, truncation=True).to(device)
            feats = self.model.get_text_features(**inputs)
            feats = self.text_proj(feats)
            text_feats.append(feats.cpu())
        text_feats = torch.cat(text_feats, dim=0)

        img_feats = None
        if image_paths is not None:
            imgs = []
            img_feats_list = []
            for p in image_paths:
                if p and Path(p).exists():
                    try:
                        imgs.append(Image.open(p).convert("RGB"))
                    except Exception:
                        imgs.append(None)
                else:
                    imgs.append(None)

            for i in range(0, len(imgs), batch_size):
                batch = imgs[i : i + batch_size]
                # replace None with a small black image to keep shapes
                batch_proc = [im if im is not None else Image.new("RGB", (224, 224), (0, 0, 0)) for im in batch]
                inputs = self.processor(images=batch_proc, return_tensors="pt", padding=True).to(device)
                feats = self.model.get_image_features(**inputs)
                feats = self.img_proj(feats)
                img_feats_list.append(feats.cpu())
            img_feats = torch.cat(img_feats_list, dim=0)

        # fusion: concat (text, image if present) -> fusion head
        if img_feats is None:
            fused = self.fusion(torch.cat([text_feats, torch.zeros_like(text_feats)], dim=1))
        else:
            fused = self.fusion(torch.cat([text_feats, img_feats], dim=1))

        # L2 normalize
        fused = nn.functional.normalize(fused, p=2, dim=1)
        return fused.numpy()


def build_faiss_index(embeddings: np.ndarray, nlist: int = 100, metric: int = faiss.METRIC_INNER_PRODUCT if faiss else None):
    if faiss is None:
        raise RuntimeError("faiss not installed; install faiss-cpu or faiss-gpu")
    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(embeddings)
    return index


def read_meta_jsonl(path: Path, text_field: str, image_field: Optional[str] = None):
    texts = []
    images = [] if image_field else None
    ids = []
    with open(path, "r", encoding="utf8") as f:
        for line in f:
            j = json.loads(line)
            ids.append(j.get("item_id") or j.get("asin") or len(ids))
            texts.append(j.get(text_field, ""))
            if image_field:
                images.append(j.get(image_field))
    return ids, texts, images


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta", type=str, help="path to meta jsonl (one JSON per line)")
    parser.add_argument("--text-field", type=str, default="title")
    parser.add_argument("--image-field", type=str, default=None)
    parser.add_argument("--out-dir", type=str, default="/tmp/clip_emb")
    parser.add_argument("--proj-dim", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ids, texts, images = read_meta_jsonl(Path(args.meta), args.text_field, args.image_field)
    model = FusionEmbedder(proj_dim=args.proj_dim, device=args.device)
    emb = model.encode(texts, images)
    np.save(out_dir / "item_ids.npy", np.array(ids))
    np.save(out_dir / "item_embeddings.npy", emb)
    print(f"Saved {len(ids)} embeddings to {out_dir}")

    if faiss is not None:
        idx = build_faiss_index(emb)
        faiss.write_index(idx, str(out_dir / "faiss_index.idx"))
        print("Built and saved FAISS index")


if __name__ == "__main__":
    main()
