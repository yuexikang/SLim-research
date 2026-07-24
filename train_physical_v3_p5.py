# 作用：以完全相同的冻结HIMO/P5、数据和损失训练Pointwise-P5与
# RectConv-P5局部描述子，用于比较是否需要矩形空间聚合。

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger

from src.physical.data import PhysicalV0DataModule
from src.physical.v3_ablation import select_gpu_with_most_free_memory
from src.physical.v3_p5_lightning import PhysicalV3P5Module
from src.physical.v3_p5_models import PhysicalV3P5Encoder


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=PhysicalV3P5Encoder.MODEL_NAMES,
        required=True,
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--train_manifest",
        type=Path,
        default=Path(
            "data/remote_archive/manifests/train_GoogleEarth_single.jsonl"
        ),
    )
    parser.add_argument(
        "--val_manifest",
        type=Path,
        default=Path(
            "data/remote_archive/manifests/val_GoogleEarth_single.jsonl"
        ),
    )
    parser.add_argument("--selected_train_rows", type=Path, default=None)
    parser.add_argument("--train_data_ratio", type=float, default=0.01)
    parser.add_argument("--max_train_rows", type=int, default=0)
    parser.add_argument("--max_val_rows", type=int, default=40)
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--effective_batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--max_epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=66)
    parser.add_argument("--task_name", default="physical_v3_p5_pilot")
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--limit_train_batches", type=int, default=0)
    parser.add_argument("--limit_val_batches", type=int, default=0)
    return parser.parse_args()


def resolve_device(value):
    if value == "auto":
        index, snapshot = select_gpu_with_most_free_memory()
        return index, snapshot
    index = int(str(value).replace("cuda:", "").rstrip(","))
    return index, None


def save_metadata(path, args, device_index, snapshot, data_module, model):
    paper = path / "paper_logs"
    paper.mkdir(parents=True, exist_ok=True)
    payload = {
        **{
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "physical_gpu": device_index,
        "gpu_snapshot": snapshot,
        "selected_base_rows": len(data_module.selected_indices),
        "train_pairs_per_epoch": len(data_module.train_dataset),
        "val_pairs_per_epoch": len(data_module.val_dataset),
        "trainable_parameters": model.encoder.trainable_parameters,
        "command": " ".join([sys.executable, *sys.argv]),
        "torch": torch.__version__,
        "pytorch_lightning": pl.__version__,
    }
    (paper / "config.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def main():
    args = parse_args()
    pl.seed_everything(args.seed, workers=True)
    torch.set_float32_matmul_precision("high")
    device_index, snapshot = resolve_device(args.device)
    accumulation = args.effective_batch_size // args.batch_size
    if args.batch_size * accumulation != args.effective_batch_size:
        raise ValueError("effective_batch_size must be divisible by batch_size")

    run_name = args.run_name or (
        f"{args.model}_googleearth_ratio1_img512_bs{args.batch_size}_"
        f"ebs{args.effective_batch_size}_seed{args.seed}_ep{args.max_epochs}"
    )
    tensorboard = TensorBoardLogger(
        save_dir="logs/tb_logs",
        name=args.task_name,
        version=run_name,
        default_hp_metric=False,
    )
    experiment_dir = Path(tensorboard.log_dir)
    csv_logger = CSVLogger(
        save_dir=str(experiment_dir / "paper_logs"),
        name="metrics",
        version="",
    )
    data_module = PhysicalV0DataModule(
        train_manifest=args.train_manifest,
        val_manifest=args.val_manifest,
        experiment_dir=experiment_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        val_batch_size=1,
        num_workers=args.num_workers,
        train_data_ratio=args.train_data_ratio,
        max_train_rows=args.max_train_rows,
        max_val_rows=args.max_val_rows,
        homography_difficulty=0.7,
        seed=args.seed,
        selected_train_rows=args.selected_train_rows,
        train_one_variant_per_row=True,
        val_one_variant_per_row=True,
        rotation_limit_degrees=45.0,
        minimum_region_sampler=True,
        photometric_augmentation="lg",
        valid_crop_rectification=True,
    )
    data_module.setup("fit")
    model = PhysicalV3P5Module(
        model_name=args.model,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_epochs=args.max_epochs,
    )
    save_metadata(
        experiment_dir,
        args,
        device_index,
        snapshot,
        data_module,
        model,
    )

    checkpoint_dir = experiment_dir / "checkpoints"
    best = ModelCheckpoint(
        dirpath=str(checkpoint_dir),
        filename="best-{epoch:02d}-{val_r0:.4f}",
        monitor="val_r0",
        mode="max",
        save_top_k=1,
        save_last=True,
        auto_insert_metric_name=False,
    )
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=[device_index],
        logger=[tensorboard, csv_logger],
        callbacks=[best, LearningRateMonitor(logging_interval="epoch")],
        max_epochs=args.max_epochs,
        precision="bf16-mixed",
        gradient_clip_val=1.0,
        accumulate_grad_batches=accumulation,
        limit_train_batches=(
            args.limit_train_batches if args.limit_train_batches > 0 else 1.0
        ),
        limit_val_batches=(
            args.limit_val_batches if args.limit_val_batches > 0 else 1.0
        ),
        num_sanity_val_steps=0,
        log_every_n_steps=1,
        deterministic=False,
    )
    print(
        f"model={args.model}, physical_gpu={device_index}, "
        f"parameters={model.encoder.trainable_parameters}, "
        f"train_rows={len(data_module.selected_indices)}, "
        f"train_pairs={len(data_module.train_dataset)}, "
        f"val_pairs={len(data_module.val_dataset)}"
    )
    trainer.fit(model, datamodule=data_module)
    summary = {
        "model": args.model,
        "best_checkpoint": best.best_model_path,
        "best_val_r0": float(best.best_model_score or 0.0),
        "last_checkpoint": best.last_model_path,
    }
    output = experiment_dir / "paper_logs" / "training_summary.json"
    output.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
