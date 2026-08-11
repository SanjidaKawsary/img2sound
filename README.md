# Img2Soundscape — Server Setup

## Directory Structure

```
img_2_sound/
├── 0.py                        # Utility: delete _orig/_recon files
├── 1audio_sonicUrban.py        # Step 1: audio → VAE embeddings (.npy)
├── 2image_sonicUrban.py        # Step 2: image → ImageBind embeddings (.npy)
├── 3encode.sh                  # Convenience wrapper for steps 1 & 2
├── 4train.py                   # Step 3: train diffusion model
├── 5inference.py               # Step 4a: inference (with audio reference)
├── 5inference_onlyimage.py     # Step 4b: inference (image only)
├── setup_and_run.sh            # Master pipeline script
├── requirements.txt
├── data/
│   ├── audios/                 # Place your .wav files here
│   └── images/                 # Place matching .jpg/.png files here
├── embeddings/
│   ├── audio/                  # Generated audio embeddings
│   └── image/                  # Generated image embeddings
├── models/
│   └── stable-audio-open/      # Downloaded VAE model cache
├── results/
│   ├── checkpoints/            # Saved model checkpoints
│   └── logs/                   # TensorBoard logs
├── generated_audio/            # Inference output .wav files
├── dataset/                    # Dataset classes
├── diffusion/                  # Diffusion model code
├── encoder/                    # ImageBind & audio encoder
└── third_party/                # Third-party utilities
```

## Quick Start

### 1. Set environment variables
```bash
export HF_TOKEN="hf_YOUR_TOKEN_HERE"
```

### 2. Accept model terms
Go to https://huggingface.co/stabilityai/stable-audio-open-1.0 and accept the terms.

### 3. Place your data
Put matched audio/image pairs in `data/audios/` and `data/images/` 
(filenames must match, e.g., `scene_001.wav` ↔ `scene_001.jpg`).

### 4. Run everything
```bash
cd /lstr/sahara/datalab-ml/ibrahim/limagents_update/urp/img_2_sound
bash setup_and_run.sh
```

### Or run individual steps
```bash
bash setup_and_run.sh --step 3   # start from audio embeddings (skip install + weights)
bash setup_and_run.sh --step 5   # just train
bash setup_and_run.sh --step 6   # just inference
```

### Or run scripts directly
```bash
# Audio embeddings
python 1audio_sonicUrban.py --input-dir data/audios --output-dir embeddings/audio

# Image embeddings
python 2image_sonicUrban.py --input-dir data/images --output-dir embeddings/image

# Train (small batch for testing)
python 4train.py \
    --image-embedding-root-path embeddings/image \
    --audio-embedding-root-path embeddings/audio \
    --work-dir results \
    --batch-size 4 --num-epochs 30

# Inference
python 5inference.py \
    --trained-model-path results/checkpoints/epoch_30_step_XXXX.pth \
    --output-dir generated_audio
```

## Key Changes from Original

1. **All hardcoded paths** → server base path or argparse defaults
2. **HF token** → read from `--hf-token` flag or `HF_TOKEN` env var (not hardcoded)
3. **ImageBind weights** → auto-resolved from `IMAGEBIND_WEIGHTS` env var or local `.checkpoints/` dir
4. **`--resume-from`** → defaults to `None` (train from scratch) instead of a specific checkpoint
5. **pytorchvideo fix** → handled in `setup_and_run.sh`
