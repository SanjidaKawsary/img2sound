import os
import glob
import torch
import torchaudio
import numpy as np
import argparse
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from diffusers import AutoencoderOobleck
from typing import List
from tqdm import tqdm

torchaudio.set_audio_backend("soundfile")

# ---------------------------------------------------------------------------
# Paths & constants  (all configurable via argparse / env vars)
# ---------------------------------------------------------------------------
BASE_DIR      = "img_2_sound"
REPO_ID       = "stabilityai/stable-audio-open-1.0"
SUBFOLDER     = "vae"
TARGET_SR     = 44100
TARGET_CH     = 2
BATCH_SIZE    = 6


def ensure_channels(x: torch.Tensor, target_c: int) -> torch.Tensor:
    """x: [C, T] -> [target_c, T]"""
    C, T = x.shape
    if C == target_c:
        return x
    if C > target_c:
        return x.mean(dim=0, keepdim=True).repeat(target_c, 1)
    return x.repeat(target_c // C if target_c % C == 0 else target_c, 1)[:target_c]


def load_wav_processed(fp: str):
    """Read -> resample to TARGET_SR -> align channels to TARGET_CH -> clamp [-1,1]"""
    x, sr = torchaudio.load(fp)
    seconds = x.shape[-1] / float(sr)
    if sr != TARGET_SR:
        x = torchaudio.functional.resample(x, sr, TARGET_SR)
    x = ensure_channels(x, TARGET_CH).clamp(-1, 1)
    return x, sr, seconds


def collate_pad_to_max(batch_tensors: List[torch.Tensor]) -> torch.Tensor:
    """Pad a list of [C,T] tensors to the same length -> [B, C, T_max]"""
    max_T = max(t.shape[-1] for t in batch_tensors)
    padded = []
    for t in batch_tensors:
        pad = max_T - t.shape[-1]
        if pad > 0:
            t = F.pad(t, (0, pad))
        padded.append(t.unsqueeze(0))
    return torch.cat(padded, dim=0)


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    hf_token = args.hf_token or os.environ.get("HF_TOKEN", None)

    # Download VAE weights if not already cached
    model_dir = args.model_dir
    snapshot_download(
        repo_id=REPO_ID,
        allow_patterns=[f"{SUBFOLDER}/*"],
        local_dir=model_dir,
        local_dir_use_symlinks=False,
        token=hf_token,
    )

    # Load VAE encoder
    vae = AutoencoderOobleck.from_pretrained(
        model_dir, subfolder=SUBFOLDER, token=hf_token, use_safetensors=True
    ).to(device).eval()

    # Collect input WAVs
    wavs = sorted(glob.glob(os.path.join(args.input_dir, "**", "*.wav"), recursive=True))
    if not wavs:
        print("No WAV files found in", args.input_dir)
        return
    print(f"Found {len(wavs)} WAV files")

    os.makedirs(args.output_dir, exist_ok=True)

    # Skip already-processed files
    existing = {os.path.splitext(f)[0] for f in os.listdir(args.output_dir) if f.endswith(".npy")}
    if existing:
        wavs = [fp for fp in wavs if os.path.splitext(os.path.basename(fp))[0] not in existing]
        print(f"Skipping {len(existing)} already processed, {len(wavs)} remaining")

    batch_size = args.batch_size
    for i in tqdm(range(0, len(wavs), batch_size), desc="Encoding audio", unit="batch"):
        batch_files = wavs[i : i + batch_size]

        xs_proc, valid_files = [], []
        for fp in batch_files:
            try:
                x_proc, _, _ = load_wav_processed(fp)
                xs_proc.append(x_proc)
                valid_files.append(fp)
            except Exception as e:
                print(f"[ERROR-READ] {fp}: {e}")

        if not xs_proc:
            continue

        x_batch = collate_pad_to_max(xs_proc).to(device)

        with torch.no_grad():
            enc = vae.encode(x_batch)
            z = enc.latent_dist.mean if hasattr(enc, "latent_dist") else getattr(enc, "latents", enc)

        for j, fp in enumerate(valid_files):
            fn = os.path.splitext(os.path.basename(fp))[0] + ".npy"
            out_fp = os.path.join(args.output_dir, fn)
            np.save(out_fp, z[j].cpu().numpy())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate audio embeddings via Stable Audio VAE")
    parser.add_argument("--input-dir", type=str,
                        default=os.path.join(BASE_DIR, "data", "audios"),
                        help="Directory with input .wav files")
    parser.add_argument("--output-dir", type=str,
                        default=os.path.join(BASE_DIR, "embeddings", "audio"),
                        help="Directory to save .npy audio embeddings")
    parser.add_argument("--model-dir", type=str,
                        default=os.path.join(BASE_DIR, "models", "stable-audio-open"),
                        help="Local cache directory for the Stable Audio VAE model")
    parser.add_argument("--hf-token", type=str, default=None,
                        help="HuggingFace token (or set HF_TOKEN env var)")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()
    main(args)
