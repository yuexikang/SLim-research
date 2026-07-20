# 作用：独立训练 Physical Encoder V0 或参数量匹配的 Tiny CNN，只使用单图在线 Homography 与 P-P 粗匹配监督。

import argparse
import json
import os
import sys
from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy

from src.physical.data import PhysicalV0DataModule
from src.physical.lightning import PhysicalV0Module
from src.physical.models import build_physical_v0_encoder, count_trainable_parameters
from utils.misc import setup_gpus


def parse_limit(value):
    text = str(value).strip()
    if text.isdigit():
        return int(text)
    return float(text)


def parse_args():
    parser = argparse.ArgumentParser(description="Train the standalone Physical Encoder V0 experiment.")
    parser.add_argument(
        "--model",
        choices=["physical_full", "tiny_cnn", "physical_no_canon", "physical_single_scale"],
        default="physical_full",
    )
    parser.add_argument("--device", default="0,", help='Physical GPU ids, for example "1," or "1,2,".')
    parser.add_argument("--train_manifest", type=Path, default=Path("data/remote_archive/manifests/train_optical_single_images.jsonl"))
    parser.add_argument("--val_manifest", type=Path, default=Path("data/remote_archive/manifests/val_optical_single_images.jsonl"))
    parser.add_argument("--train_data_ratio", type=float, default=0.3)
    parser.add_argument("--max_train_rows", type=int, default=0, help="Limit selected base rows after ratio sampling; 0 disables the limit.")
    parser.add_argument("--max_val_rows", type=int, default=0, help="Limit validation base rows; 0 uses all rows.")
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--val_batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=6)
    parser.add_argument("--max_epochs", type=int, default=20)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--homography_difficulty", type=float, default=0.3)
    parser.add_argument("--similarity_mode", choices=["full", "chunked"], default="full")
    parser.add_argument("--chunk_size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=66)
    parser.add_argument("--task_name", default="physical_v0")
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--resume_ckpt_path", type=Path, default=None)
    parser.add_argument("--save_every_n_epochs", type=int, default=2)
    parser.add_argument("--limit_train_batches", type=parse_limit, default=1.0)
    parser.add_argument("--limit_val_batches", type=parse_limit, default=1.0)
    parser.add_argument("--num_sanity_val_steps", type=int, default=0)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", default="slim_physical_v0")
    parser.add_argument("--wandb_entity", default=None)
    parser.add_argument("--wandb_mode", choices=["online", "offline", "disabled"], default="offline")
    parser.add_argument("--wandb_log_model", choices=["all", "true", "false"], default="false")
    return parser.parse_args()


def wandb_log_model(value):
    if value == "all":
        return "all"
    return value == "true"


def save_metadata(experiment_dir, args, device_count, parameter_counts, data_module):
    paper_dir = experiment_dir / "paper_logs"
    paper_dir.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy()
    config = {key: str(value) if isinstance(value, Path) else value for key, value in config.items()}
    config.update(
        {
            "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "device_count": device_count,
            "parameter_counts": parameter_counts,
            "selected_base_rows": len(data_module.selected_indices),
            "train_pairs_per_epoch": len(data_module.train_dataset),
            "val_pairs": len(data_module.val_dataset),
        }
    )
    (paper_dir / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (paper_dir / "command.txt").write_text(
        " ".join([sys.executable, *sys.argv]) + "\n", encoding="utf-8"
    )
    environment = [
        f"python: {sys.version.split()[0]}",
        f"torch: {torch.__version__}",
        f"pytorch_lightning: {pl.__version__}",
        f"cuda_visible_devices: {os.environ.get('CUDA_VISIBLE_DEVICES', '')}",
        f"cuda_device_count: {torch.cuda.device_count()}",
    ]
    (paper_dir / "environment.txt").write_text("\n".join(environment) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    if args.image_size % 8:
        raise ValueError("image_size must be divisible by 8.")
    pl.seed_everything(args.seed, workers=True)
    torch.set_float32_matmul_precision("high")
    device_count = setup_gpus(args.device)
    if device_count < 1 or not torch.cuda.is_available():
        raise RuntimeError("Physical V0 training requires at least one CUDA GPU.")

    parameter_counts = {
        name: count_trainable_parameters(build_physical_v0_encoder(name))
        for name in ("physical_full", "tiny_cnn", "physical_no_canon", "physical_single_scale")
    }
    parameter_gap = abs(parameter_counts["physical_full"] - parameter_counts["tiny_cnn"]) / parameter_counts["physical_full"]
    if parameter_gap > 0.05:
        raise RuntimeError(f"Tiny CNN parameter gap is {parameter_gap:.2%}, exceeding 5%.")

    run_name = args.run_name or f"{args.model}_seed{args.seed}"
    tensorboard = TensorBoardLogger(
        save_dir="logs/tb_logs",
        name=args.task_name,
        version=run_name,
        default_hp_metric=False,
    )
    experiment_dir = Path(tensorboard.log_dir)
    csv_logger = CSVLogger(
        save_dir=str(experiment_dir / "paper_logs"), name="metrics", version=""
    )
    loggers = [tensorboard, csv_logger]
    if args.use_wandb:
        try:
            from pytorch_lightning.loggers import WandbLogger
        except ImportError as exc:
            raise ImportError("Install wandb or run without --use_wandb.") from exc
        loggers.append(
            WandbLogger(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=run_name,
                save_dir=str(experiment_dir / "wandb"),
                mode=args.wandb_mode,
                log_model=wandb_log_model(args.wandb_log_model),
                tags=["physical_v0", args.model],
            )
        )

    data_module = PhysicalV0DataModule(
        train_manifest=args.train_manifest,
        val_manifest=args.val_manifest,
        experiment_dir=experiment_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        val_batch_size=args.val_batch_size,
        num_workers=args.num_workers,
        train_data_ratio=args.train_data_ratio,
        max_train_rows=args.max_train_rows,
        max_val_rows=args.max_val_rows,
        homography_difficulty=args.homography_difficulty,
        seed=args.seed,
    )
    data_module.setup("fit")
    save_metadata(experiment_dir, args, device_count, parameter_counts, data_module)

    model = PhysicalV0Module(
        model_name=args.model,
        similarity_mode=args.similarity_mode,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_epochs=args.max_epochs,
        chunk_size=args.chunk_size,
    )
    for logger in loggers:
        logger.log_hyperparams(
            {
                **{key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
                **parameter_counts,
                "selected_base_rows": len(data_module.selected_indices),
                "train_pairs_per_epoch": len(data_module.train_dataset),
            }
        )

    checkpoint_dir = experiment_dir / "checkpoints"
    callbacks = [
        ModelCheckpoint(
            dirpath=str(checkpoint_dir),
            filename="best-{epoch:02d}-{val_r1:.4f}",
            monitor="val_r1",
            mode="max",
            save_top_k=1,
            save_last=True,
            auto_insert_metric_name=False,
            verbose=True,
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]
    if args.save_every_n_epochs > 0:
        callbacks.append(
            ModelCheckpoint(
                dirpath=str(checkpoint_dir / "epoch_checkpoints"),
                filename="epoch-{epoch:03d}",
                every_n_epochs=args.save_every_n_epochs,
                save_top_k=-1,
                auto_insert_metric_name=False,
            )
        )

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=device_count,
        strategy=(
            DDPStrategy(find_unused_parameters=False)
            if device_count > 1
            else "auto"
        ),
        logger=loggers,
        callbacks=callbacks,
        max_epochs=args.max_epochs,
        # BF16 keeps FP32-like exponent range, which avoids GradScaler skipping
        # the first coarse dual-softmax updates on RTX 4090 GPUs.
        precision="bf16-mixed" if args.amp else "32-true",
        gradient_clip_val=1.0,
        limit_train_batches=args.limit_train_batches,
        limit_val_batches=args.limit_val_batches,
        num_sanity_val_steps=args.num_sanity_val_steps,
        check_val_every_n_epoch=1,
        log_every_n_steps=1 if isinstance(args.limit_train_batches, int) else 50,
        deterministic=False,
    )
    print(
        f"Model={args.model}, parameters={parameter_counts[args.model]:,}, "
        f"selected rows={len(data_module.selected_indices):,}, "
        f"pairs/epoch={len(data_module.train_dataset):,}, val pairs={len(data_module.val_dataset):,}"
    )
    trainer.fit(model, datamodule=data_module, ckpt_path=args.resume_ckpt_path)


if __name__ == "__main__":
    main()
