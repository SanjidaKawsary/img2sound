import argparse
import os
import warnings
import torch
import torchaudio
from tqdm import tqdm
from diffusers import AutoencoderOobleck

from diffusion import IDDPM
from diffusion.model.builder import build_model
from diffusion.utils.checkpoint import load_checkpoint
from diffusion.data.builder import build_dataloader
from dataset.dataset import InferenceDataset2

warnings.filterwarnings("ignore")

BASE_DIR = "img_2_sound"


@torch.inference_mode()
def soundlize(model, sample_steps, image_embedding, device=None):
    """Generate audio from image embedding only (no reference audio needed)."""
    bs = image_embedding.shape[0]
    T, dim = 64, 215  # fixed latent shape
    z = torch.randn(bs, T, dim, device=device)
    model_kwargs = dict(image_embedding=image_embedding)
    diffusion = IDDPM(str(sample_steps))
    samples = diffusion.p_sample_loop(
        model.forward, z.shape, z, clip_denoised=False,
        model_kwargs=model_kwargs, progress=True, device=device,
    )
    return samples


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    hf_token = args.hf_token or os.environ.get("HF_TOKEN", None)

    dataset = InferenceDataset2(
        image_embeddings_root=args.image_embedding_root_path,
        output_dir=args.output_dir,
    )
    dataloader = build_dataloader(dataset, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vae = AutoencoderOobleck.from_pretrained(
        args.vae_model_path, subfolder=args.vae_subfolder,
        token=hf_token, use_safetensors=True,
    ).to(device).eval()

    image_embedding_size = 1024
    model = build_model(
        "DiT_builder",
        use_grad_checkpoint=True,
        use_fp32_attention=True,
        image_embedding_size=image_embedding_size,
    )
    load_checkpoint(args.trained_model_path, model)
    model = model.to(device).eval()

    for batch in tqdm(dataloader, desc="Inference (image-only)"):
        img, name = batch
        img = img.to(device)

        with torch.no_grad():
            samples = soundlize(
                model, sample_steps=args.num_denoising_timesteps,
                image_embedding=img, device=device,
            )
            dec = vae.decode(samples)
            y_batch = dec.sample if hasattr(dec, "sample") else dec

        for i in range(y_batch.shape[0]):
            output_path = os.path.join(args.output_dir, name[i] + args.output_suffix)
            torchaudio.save(output_path, y_batch[i].cpu(), args.target_audio_sr)
            print(f"Saved: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Img2Soundscape Inference (image-only, no audio reference)")

    # VAE
    parser.add_argument("--vae-model-path", type=str,
                        default=os.path.join(BASE_DIR, "models", "stable-audio-open"))
    parser.add_argument("--vae-subfolder", type=str, default="vae")
    parser.add_argument("--hf-token", type=str, default=None,
                        help="HuggingFace token (or set HF_TOKEN env var)")
    parser.add_argument("--target-audio-sr", type=int, default=44100)

    # Data paths
    parser.add_argument("--image-embedding-root_path", type=str,
                        default=os.path.join(BASE_DIR, "embeddings", "image"),
                        dest="image_embedding_root_path")
    parser.add_argument("--trained-model-path", type=str,
                        default=os.path.join(BASE_DIR, "results", "checkpoints", "latest.pth"))

    # Output
    parser.add_argument("--output-dir", type=str,
                        default=os.path.join(BASE_DIR, "generated_audio"))
    parser.add_argument("--output-suffix", type=str, default=".wav")

    # Inference params
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-denoising-timesteps", type=int, default=100)
    parser.add_argument("--cfg-scale-image", type=float, default=4.0)

    args = parser.parse_args()
    main(args)
