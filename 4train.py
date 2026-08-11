import argparse
import datetime
import os
import sys
import time
import types
import warnings
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn as nn
from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.utils import DataLoaderConfiguration
from accelerate.utils import DistributedType
from torch.utils.data import RandomSampler

from diffusion import IDDPM
from diffusion.model.builder import build_model
from diffusion.utils.checkpoint import save_checkpoint, load_checkpoint
from diffusion.utils.data_sampler import AspectRatioBatchSampler, BalancedAspectRatioBatchSampler
from diffusion.utils.dist_utils import get_world_size, clip_grad_norm_
from diffusion.utils.logger import get_root_logger
from diffusion.utils.lr_scheduler import build_lr_scheduler
from diffusion.utils.misc import set_random_seed, read_config, init_random_seed, DebugUnderflowOverflow
from diffusion.utils.optimizer import auto_scale_lr
from diffusion.data.builder import build_dataset, build_dataloader, set_data_root

from dataset.dataset import TrainingDataset

warnings.filterwarnings("ignore")

BASE_DIR = "img_2_sound"


class LogBuffer:
    def __init__(self):
        self.logs = {}

    def update(self, logs):
        for key, value in logs.items():
            if key not in self.logs:
                self.logs[key] = []
            self.logs[key].append(value)

    def average(self):
        return {key: sum(values) / len(values) for key, values in self.logs.items()}

    def clear(self):
        self.logs = {}

    @property
    def output(self):
        return self.average()


def ema_update(model_dest: nn.Module, model_src: nn.Module, rate):
    param_dict_src = dict(model_src.named_parameters())
    for p_name, p_dest in model_dest.named_parameters():
        p_src = param_dict_src[p_name]
        assert p_src is not p_dest
        p_dest.data.mul_(rate).add_((1 - rate) * p_src.data)


def main(args):
    os.makedirs(args.work_dir, exist_ok=True)
    init_handler = InitProcessGroupKwargs()
    init_handler.timeout = datetime.timedelta(seconds=5400)
    fsdp_plugin = None
    dataloader_config = DataLoaderConfiguration(even_batches=True)

    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=max(1, args.grad_accum_steps),
        log_with="tensorboard",
        project_dir=os.path.join(args.work_dir, "logs"),
        fsdp_plugin=fsdp_plugin,
        dataloader_config=dataloader_config,
        kwargs_handlers=[init_handler],
    )
    logger = get_root_logger(os.path.join(args.work_dir, "train_log.log"))
    logger.info(f"Gradient accumulation steps: {max(1, args.grad_accum_steps)}")
    logger.info(f"Mixed precision: {args.mixed_precision}")

    seed = init_random_seed(args.seed)
    set_random_seed(seed)

    logger.info(f"World_size: {get_world_size()}, seed: {args.seed}")

    train_diffusion = IDDPM(str(args.train_sampling_steps), learn_sigma=True, pred_sigma=True, snr=False)
    image_embedding_size = 1024
    model_kwargs = {"image_embedding_size": image_embedding_size}
    model = build_model(
        "DiT_builder",
        use_grad_checkpoint=args.use_grad_checkpoint,
        use_fp32_attention=args.use_fp32_attention,
        gc_step=args.gc_step,
        **model_kwargs,
    ).train()

    logger.info(f"Model Parameters: {sum(p.numel() for p in model.parameters()):,}")
    model_ema = deepcopy(model).eval()

    dataset = TrainingDataset(
        image_embeddings_root=args.image_embedding_root_path,
        audio_embeddings_root=args.audio_embedding_root_path,
    )
    lr_scale_ratio = 1
    train_dataloader = build_dataloader(dataset, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=3e-2, eps=1e-10)
    start_epoch = 0

    lr_scheduler = build_lr_scheduler(
        config=None, optimizer=optimizer, train_dataloader=train_dataloader, lr_scale_ratio=lr_scale_ratio
    )

    ema_update(model_ema, model, 0.0)
    if accelerator.distributed_type == DistributedType.FSDP:
        for m in accelerator._models:
            m.clip_grad_norm_ = types.MethodType(clip_grad_norm_, m)
    model, model_ema = accelerator.prepare(model, model_ema)
    optimizer, train_dataloader, lr_scheduler = accelerator.prepare(optimizer, train_dataloader, lr_scheduler)

    last_path = None
    if args.resume_from is not None:
        start_epoch, missing, unexpected = load_checkpoint(
            args.resume_from,
            model=model,
            model_ema=model_ema,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
        )
        logger.warning(f"Unexpected keys: {unexpected}")
        last_path = args.resume_from

    time_start, last_tic = time.time(), time.time()
    log_buffer = LogBuffer()

    start_step = start_epoch * len(train_dataloader)
    global_step = 0
    total_steps = len(train_dataloader) * args.num_epochs

    for epoch in range(start_epoch + 1, args.num_epochs + 1):
        data_time_start = time.time()
        data_time_all = 0

        for step, batch in enumerate(train_dataloader):
            data_time_all += time.time() - data_time_start
            image, audio = batch
            additional_input = {"image_embedding": image}
            timesteps = torch.randint(0, args.train_sampling_steps, (image.size(0),), device=audio.device).long()
            grad_norm = None
            with accelerator.accumulate(model):
                loss_term = train_diffusion.training_losses(model, audio, timesteps, model_kwargs=additional_input)
                loss = loss_term["loss"].mean()
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    grad_norm = accelerator.clip_grad_norm_(model.parameters(), args.gradient_clip)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    ema_update(model_ema, model, args.ema_rate)

            lr = lr_scheduler.get_last_lr()[0]
            logs = {args.loss_report_name: accelerator.gather(loss).mean().item(), "lr": lr}
            if grad_norm is not None:
                logs.update(grad_norm=accelerator.gather(grad_norm).mean().item())

            log_buffer.update(logs)
            global_step += 1

            if (step + 1) % args.log_interval == 0 or (step + 1) == 1:
                t = (time.time() - last_tic) / args.log_interval
                t_d = data_time_all / args.log_interval
                avg_time = (time.time() - time_start) / global_step
                eta = str(datetime.timedelta(seconds=int(avg_time * (total_steps - start_step - global_step))))
                eta_epoch = str(datetime.timedelta(seconds=int(avg_time * (len(train_dataloader) - step - 1))))
                _ = log_buffer.average()
                info = (
                    f"Step/Epoch [{(epoch-1)*len(train_dataloader)+step+1}/{epoch}]"
                    f"[{step + 1}/{len(train_dataloader)}]: total_eta: {eta}, "
                    f"epoch_eta: {eta_epoch}, time_all: {t:.3f}, time_data: {t_d:.3f}, lr: {lr:.3e}), "
                )
                info += ", ".join([f"{k}: {v:.4f}" for k, v in log_buffer.output.items()])
                logger.info(info)
                last_tic = time.time()
                log_buffer.clear()
                data_time_all = 0

            accelerator.log(logs, step=global_step + start_step)
            data_time_start = time.time()

        # Save checkpoint every N epochs
        if epoch % args.save_model_epochs == 0 or epoch == args.num_epochs:
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                os.umask(0o000)
                if args.delete_previous_checkpoint != 1:
                    last_path = None
                last_path = save_checkpoint(
                    os.path.join(args.work_dir, "checkpoints"),
                    epoch=epoch,
                    step=(epoch - 1) * len(train_dataloader) + step + 1,
                    model=accelerator.unwrap_model(model),
                    model_ema=model_ema,
                    optimizer=optimizer,
                    lr_scheduler=lr_scheduler,
                    last_path=last_path,
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Img2Soundscape diffusion model")

    # Working directory for logs & checkpoints
    parser.add_argument("--work-dir", default=os.path.join(BASE_DIR, "results"),
                        help="Directory for logs and checkpoints")
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--train-sampling-steps", type=int, default=1000)

    # Gradient / precision
    parser.add_argument("--grad-accum-steps", type=int, default=0)
    parser.add_argument("--use-grad-checkpoint", action="store_true", default=False)
    parser.add_argument("--use-fp32-attention", action="store_true", default=False)
    parser.add_argument("--gc-step", type=int, default=1)
    parser.add_argument("--mixed-precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])

    # Training
    parser.add_argument("--resume-from", default=None,
                        help="Path to checkpoint .pth to resume from (None = train from scratch)")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-epochs", type=int, default=200)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--ema-rate", type=float, default=0.9999)

    # Logging / saving
    parser.add_argument("--loss-report-name", type=str, default="loss")
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--save-model-steps", type=int, default=10000)
    parser.add_argument("--save-model-epochs", type=int, default=1)
    parser.add_argument("--delete-previous-checkpoint", type=int, default=1)

    # Dataset paths
    parser.add_argument("--image-embedding-root-path", type=str,
                        default=os.path.join(BASE_DIR, "embeddings", "image"))
    parser.add_argument("--audio-embedding-root-path", type=str,
                        default=os.path.join(BASE_DIR, "embeddings", "audio"))

    args = parser.parse_args()
    main(args)
