# 作用：训练Physical Encoder V2粗匹配残差分支；SLiM保持冻结，不运行Fine或Refinement，也不自动执行正式多模态测评。

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
from src.physical.v2_lightning import PhysicalV2Module
from src.physical.v2_models import PhysicalEncoderV2, build_physical_v2_encoder
from utils.misc import setup_gpus


DEFAULT_TRAIN_MANIFEST = Path(
    "data/remote_archive/manifests/train_GoogleEarth_single.jsonl"
)
DEFAULT_VAL_MANIFEST = Path(
    "data/remote_archive/manifests/val_GoogleEarth_single.jsonl"
)
IMPLEMENTATION_VERSION = "2.1.2"


def parse_limit(value):
    text = str(value).strip()
    if text.isdigit():
        return int(text)
    return float(text)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the frozen-SLiM Physical Encoder V2 coarse residual."
    )
    parser.add_argument(
        "--model",
        choices=list(PhysicalEncoderV2.MODEL_CONFIGS),
        default="physical_v2_core",
    )
    parser.add_argument("--device", default="0,")
    parser.add_argument("--train_manifest", type=Path, default=DEFAULT_TRAIN_MANIFEST)
    parser.add_argument("--val_manifest", type=Path, default=DEFAULT_VAL_MANIFEST)
    parser.add_argument(
        "--slim_checkpoint", type=Path, default=Path("ckpt/megadepth_19epochs.ckpt")
    )
    parser.add_argument(
        "--train_data_ratio",
        type=float,
        default=1.0,
        help="Fraction of the selected training manifest to use.",
    )
    parser.add_argument("--max_train_rows", type=int, default=0)
    parser.add_argument("--max_val_rows", type=int, default=0)
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=6)
    parser.add_argument("--val_batch_size", type=int, default=1)
    parser.add_argument("--effective_batch_size", type=int, default=6)
    parser.add_argument("--accumulate_grad_batches", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=6)
    parser.add_argument("--max_epochs", type=int, default=20)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--physical_learning_rate", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--homography_difficulty", type=float, default=0.3)
    parser.add_argument("--chunk_size", type=int, default=256)
    parser.add_argument("--polar_chunk_size", type=int, default=1024)
    parser.add_argument("--gradient_log_interval", type=int, default=200)
    parser.add_argument(
        "--visualize_feature_maps",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--feature_visualization_interval", type=int, default=20)
    parser.add_argument("--seed", type=int, default=66)
    parser.add_argument("--task_name", default="physical_v2_googleearth_single")
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--resume_ckpt_path", type=Path, default=None)
    parser.add_argument("--save_every_n_epochs", type=int, default=2)
    parser.add_argument("--limit_train_batches", type=parse_limit, default=1.0)
    parser.add_argument("--limit_val_batches", type=parse_limit, default=1.0)
    parser.add_argument("--num_sanity_val_steps", type=int, default=0)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--train_one_variant_per_row",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--val_one_variant_per_row",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--full_validate_best_at_end",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", default="slim_physical_v2")
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
        micro_global = args.batch_size * device_count
        if args.effective_batch_size % micro_global:
            raise ValueError(
                "effective_batch_size must be divisible by batch_size * GPU count; "
                f"got {args.effective_batch_size} and {micro_global}."
            )
        accumulation = args.effective_batch_size // micro_global
    actual = args.batch_size * device_count * accumulation
    if actual != args.effective_batch_size:
        raise ValueError(f"Configured effective batch is {actual}, expected {args.effective_batch_size}.")
    return accumulation


def wandb_log_model(value):
    return "all" if value == "all" else value == "true"


def build_run_name(args, device_count):
    subset = round(args.train_data_ratio * 100)
    return (
        f"{args.model}_googleearth_subset{subset}_img{args.image_size}_"
        f"gpu{device_count}_bs{args.batch_size}_ebs{args.effective_batch_size}_"
        f"seed{args.seed}_ep{args.max_epochs}_chunk{args.chunk_size}"
    )


def save_metadata(experiment_dir, args, device_count, accumulation, data_module, counts):
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
            "selected_base_rows": len(data_module.selected_indices),
            "train_pairs_per_epoch": len(data_module.train_dataset),
            "val_pairs_during_training": len(data_module.val_dataset),
            "full_val_pairs": len(data_module.full_val_dataset),
            "parameter_counts": counts,
            "implementation_version": IMPLEMENTATION_VERSION,
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
    if args.image_size != 512:
        raise ValueError("Physical V2 is specified and validated for image_size=512.")
    for path in (args.train_manifest, args.val_manifest, args.slim_checkpoint):
        if not path.exists():
            raise FileNotFoundError(path)
    pl.seed_everything(args.seed, workers=True)
    torch.set_float32_matmul_precision("high")
    device_count = setup_gpus(args.device)
    if device_count < 1 or not torch.cuda.is_available():
        raise RuntimeError("Physical V2 training requires at least one CUDA GPU.")
    accumulation = resolve_accumulation(args, device_count)

    counts = {
        name: count_trainable_parameters(build_physical_v2_encoder(name))
        for name in PhysicalEncoderV2.MODEL_CONFIGS
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
                    "physical_v2",
                    "physical_v2_1_2",
                    "googleearth_single",
                    args.model,
                    "one_variant_per_row",
                    f"seed{args.seed}",
                ],
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
        selected_train_rows=None,
        train_one_variant_per_row=args.train_one_variant_per_row,
        val_one_variant_per_row=args.val_one_variant_per_row,
    )
    data_module.setup("fit")
    save_metadata(experiment_dir, args, device_count, accumulation, data_module, counts)

    model = PhysicalV2Module(
        model_name=args.model,
        slim_checkpoint=str(args.slim_checkpoint),
        learning_rate=args.learning_rate,
        physical_learning_rate=args.physical_learning_rate,
        weight_decay=args.weight_decay,
        max_epochs=args.max_epochs,
        chunk_size=args.chunk_size,
        gradient_log_interval=args.gradient_log_interval,
        polar_chunk_size=args.polar_chunk_size,
        feature_visualization_interval=(
            args.feature_visualization_interval if args.visualize_feature_maps else 0
        ),
        feature_visualization_dir=str(
            experiment_dir / "paper_logs" / "latest_feature_maps"
        ),
    )
    hyperparameters = {
        **{
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "model_trainable_parameters": model.parameter_count,
        "all_ablation_parameter_counts": counts,
        "accumulate_grad_batches": accumulation,
        "selected_base_rows": len(data_module.selected_indices),
        "train_pairs_per_epoch": len(data_module.train_dataset),
        "val_pairs_during_training": len(data_module.val_dataset),
        "full_val_pairs": len(data_module.full_val_dataset),
        "implementation_version": IMPLEMENTATION_VERSION,
    }
    for logger in loggers:
        logger.log_hyperparams(hyperparameters)

    checkpoint_dir = experiment_dir / "checkpoints"
    best_checkpoint = ModelCheckpoint(
        dirpath=str(checkpoint_dir),
        filename="best-{epoch:02d}",
        monitor="val/enhanced_r0",
        mode="max",
        save_top_k=1,
        save_last=True,
        auto_insert_metric_name=False,
        verbose=True,
    )
    callbacks = [best_checkpoint, LearningRateMonitor(logging_interval="epoch")]
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
        strategy=DDPStrategy(find_unused_parameters=False) if device_count > 1 else "auto",
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
        f"Model={args.model}, trainable parameters={model.parameter_count:,}, "
        f"selected rows={len(data_module.selected_indices):,}, "
        f"pairs/epoch={len(data_module.train_dataset):,}, "
        f"val pairs/epoch={len(data_module.val_dataset):,}, "
        f"full val pairs={len(data_module.full_val_dataset):,}, accumulation={accumulation}"
    )
    trainer.fit(model, datamodule=data_module, ckpt_path=args.resume_ckpt_path)
    if not args.full_validate_best_at_end:
        return
    best_path = best_checkpoint.best_model_path
    if not best_path:
        raise RuntimeError("No best checkpoint was produced for final full validation.")
    data_module.enable_full_validation()
    trainer.limit_val_batches = 1.0
    results = trainer.validate(
        model=model,
        datamodule=data_module,
        ckpt_path=best_path,
        verbose=True,
    )
    if trainer.is_global_zero:
        output = experiment_dir / "paper_logs" / "full_validation_best.json"
        output.write_text(
            json.dumps(
                {
                    "best_checkpoint": str(Path(best_path).resolve()),
                    "selection_metric": "val/enhanced_r0",
                    "val_pairs_during_training": len(data_module.val_dataset),
                    "full_val_pairs": len(data_module.full_val_dataset),
                    "metrics": {
                        name: float(value) for name, value in (results[0] if results else {}).items()
                    },
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        print(f"Full validation summary: {output}")


if __name__ == "__main__":
    main()
