# 作用：独立训练 Physical Encoder V1-Core 及其消融模型，不加载 SLiM 权重，并记录严格 R@0 验证指标。

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
from src.physical.models import count_trainable_parameters
from src.physical.v1_lightning import PhysicalV1Module
from src.physical.v1_models import PhysicalEncoderV1, build_physical_v1_encoder
from utils.misc import setup_gpus


DEFAULT_SELECTED_ROWS = Path(
    "logs/tb_logs/physical_v0/tiny_cnn_ratio30_gpu3_bs8_seed66/selected_train_rows.jsonl"
)


def parse_limit(value):
    text = str(value).strip()
    if text.isdigit():
        return int(text)
    return float(text)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the standalone Physical Encoder V1 experiment."
    )
    parser.add_argument(
        "--model",
        choices=list(PhysicalEncoderV1.MODEL_CONFIGS),
        default="physical_v1_core",
    )
    parser.add_argument("--device", default="0,", help='Physical GPU ids, such as "3,".')
    parser.add_argument(
        "--train_manifest",
        type=Path,
        default=Path("data/remote_archive/manifests/train_optical_single_images.jsonl"),
    )
    parser.add_argument(
        "--val_manifest",
        type=Path,
        default=Path("data/remote_archive/manifests/val_optical_single_images.jsonl"),
    )
    parser.add_argument("--selected_train_rows", type=Path, default=DEFAULT_SELECTED_ROWS)
    parser.add_argument(
        "--resample_train_subset",
        action="store_true",
        help="Ignore selected_train_rows and perform a fresh stratified ratio sample.",
    )
    parser.add_argument("--train_data_ratio", type=float, default=0.3)
    parser.add_argument("--max_train_rows", type=int, default=0)
    parser.add_argument("--max_val_rows", type=int, default=0)
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=2, help="Per-GPU micro batch size.")
    parser.add_argument("--val_batch_size", type=int, default=1)
    parser.add_argument("--effective_batch_size", type=int, default=8)
    parser.add_argument(
        "--accumulate_grad_batches",
        type=int,
        default=0,
        help="0 computes accumulation from effective_batch_size.",
    )
    parser.add_argument("--num_workers", type=int, default=6)
    parser.add_argument("--max_epochs", type=int, default=20)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--physical_learning_rate", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--homography_difficulty", type=float, default=0.3)
    parser.add_argument(
        "--train_one_variant_per_row",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Sample one deterministic random perturbation per base image and epoch.",
    )
    parser.add_argument("--similarity_mode", choices=["full", "chunked"], default="chunked")
    parser.add_argument("--chunk_size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=66)
    parser.add_argument("--task_name", default="physical_v1_optical_single")
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--resume_ckpt_path", type=Path, default=None)
    parser.add_argument("--save_every_n_epochs", type=int, default=2)
    parser.add_argument("--limit_train_batches", type=parse_limit, default=1.0)
    parser.add_argument("--limit_val_batches", type=parse_limit, default=1.0)
    parser.add_argument("--num_sanity_val_steps", type=int, default=0)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", default="slim_physical_v1")
    parser.add_argument("--wandb_entity", default=None)
    parser.add_argument(
        "--wandb_mode", choices=["online", "offline", "disabled"], default="online"
    )
    parser.add_argument(
        "--wandb_log_model", choices=["all", "true", "false"], default="false"
    )
    return parser.parse_args()


def resolve_accumulation(args, device_count):
    if args.accumulate_grad_batches > 0:
        accumulation = args.accumulate_grad_batches
    else:
        micro_global_batch = args.batch_size * device_count
        if args.effective_batch_size % micro_global_batch:
            raise ValueError(
                "effective_batch_size must be divisible by batch_size * GPU count; "
                f"got {args.effective_batch_size} and {micro_global_batch}."
            )
        accumulation = args.effective_batch_size // micro_global_batch
    actual = args.batch_size * device_count * accumulation
    if actual != args.effective_batch_size:
        raise ValueError(
            f"Configured global batch is {actual}, expected {args.effective_batch_size}."
        )
    return accumulation


def wandb_log_model(value):
    if value == "all":
        return "all"
    return value == "true"


def build_run_name(args, device_count):
    ratio_percent = round(args.train_data_ratio * 100)
    return (
        f"{args.model}_optical_single_ratio{ratio_percent}_img{args.image_size}_"
        f"gpu{device_count}_bs{args.batch_size}_ebs{args.effective_batch_size}_"
        f"seed{args.seed}_ep{args.max_epochs}_{args.similarity_mode}"
    )


def save_metadata(experiment_dir, args, device_count, accumulation, parameter_counts, data_module):
    paper_dir = experiment_dir / "paper_logs"
    paper_dir.mkdir(parents=True, exist_ok=True)
    config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    config.update(
        {
            "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "device_count": device_count,
            "accumulate_grad_batches": accumulation,
            "actual_effective_batch_size": args.batch_size * device_count * accumulation,
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
    (paper_dir / "environment.txt").write_text(
        "\n".join(environment) + "\n", encoding="utf-8"
    )


def main():
    args = parse_args()
    if args.image_size % 8 or args.image_size < 64:
        raise ValueError("image_size must be at least 64 and divisible by 8.")
    pl.seed_everything(args.seed, workers=True)
    torch.set_float32_matmul_precision("high")
    device_count = setup_gpus(args.device)
    if device_count < 1 or not torch.cuda.is_available():
        raise RuntimeError("Physical V1 training requires at least one CUDA GPU.")
    accumulation = resolve_accumulation(args, device_count)

    parameter_counts = {
        name: count_trainable_parameters(build_physical_v1_encoder(name))
        for name in PhysicalEncoderV1.MODEL_CONFIGS
    }
    run_name = args.run_name or build_run_name(args, device_count)
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
                group=args.task_name,
                save_dir=str(experiment_dir / "wandb"),
                mode=args.wandb_mode,
                log_model=wandb_log_model(args.wandb_log_model),
                tags=[
                    "physical_v1",
                    "optical_single",
                    args.model,
                    f"ratio{round(args.train_data_ratio * 100)}",
                    f"seed{args.seed}",
                ],
            )
        )

    selected_rows = None if args.resample_train_subset else args.selected_train_rows
    if selected_rows is not None and not selected_rows.exists():
        raise FileNotFoundError(
            f"Fair-comparison selected rows file does not exist: {selected_rows}"
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
        selected_train_rows=selected_rows,
        train_one_variant_per_row=args.train_one_variant_per_row,
    )
    data_module.setup("fit")
    save_metadata(
        experiment_dir, args, device_count, accumulation, parameter_counts, data_module
    )

    model = PhysicalV1Module(
        model_name=args.model,
        similarity_mode=args.similarity_mode,
        learning_rate=args.learning_rate,
        physical_learning_rate=args.physical_learning_rate,
        weight_decay=args.weight_decay,
        max_epochs=args.max_epochs,
        chunk_size=args.chunk_size,
    )
    hyperparameters = {
        **{
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        **parameter_counts,
        "accumulate_grad_batches": accumulation,
        "selected_base_rows": len(data_module.selected_indices),
        "train_pairs_per_epoch": len(data_module.train_dataset),
    }
    for logger in loggers:
        logger.log_hyperparams(hyperparameters)

    checkpoint_dir = experiment_dir / "checkpoints"
    callbacks = [
        ModelCheckpoint(
            dirpath=str(checkpoint_dir),
            filename="best-{epoch:02d}",
            monitor="val/all_r0",
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
            DDPStrategy(find_unused_parameters=False) if device_count > 1 else "auto"
        ),
        logger=loggers,
        callbacks=callbacks,
        max_epochs=args.max_epochs,
        precision="bf16-mixed" if args.amp else "32-true",
        gradient_clip_val=1.0,
        accumulate_grad_batches=accumulation,
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
        f"pairs/epoch={len(data_module.train_dataset):,}, "
        f"val pairs={len(data_module.val_dataset):,}, accumulation={accumulation}"
    )
    trainer.fit(model, datamodule=data_module, ckpt_path=args.resume_ckpt_path)


if __name__ == "__main__":
    main()
