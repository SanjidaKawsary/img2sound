import os
import argparse
import numpy as np
from tqdm import tqdm
from encoder.image.image_embedder import ImageFeatureExtractor

BASE_DIR = "img_2_sound"


def main(args):
    image_encoder = ImageFeatureExtractor()

    os.makedirs(args.output_dir, exist_ok=True)

    imgs = sorted(os.listdir(args.input_dir))
    if not imgs:
        raise ValueError(f"No images found in {args.input_dir}")
    print(f"Found {len(imgs)} images")

    # Skip already-processed files
    existing = {os.path.splitext(f)[0] for f in os.listdir(args.output_dir) if f.endswith(".npy")}
    if existing:
        imgs = [f for f in imgs if os.path.splitext(f)[0] not in existing]
        print(f"Skipping {len(existing)} already processed, {len(imgs)} remaining")

    for img in tqdm(imgs, desc="Encoding images"):
        img_path = os.path.join(args.input_dir, img)
        save_path = os.path.join(args.output_dir, os.path.splitext(img)[0] + ".npy")
        image_encoder.extract_features(img_path, save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate image embeddings via ImageBind")
    parser.add_argument("--input-dir", type=str,
                        default=os.path.join(BASE_DIR, "data", "images"),
                        help="Directory with input image files")
    parser.add_argument("--output-dir", type=str,
                        default=os.path.join(BASE_DIR, "embeddings", "image"),
                        help="Directory to save .npy image embeddings")
    args = parser.parse_args()
    main(args)
