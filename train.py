import argparse
import os
import pprint
import sys
from pathlib import Path
import torch
import yaml
torch.set_float32_matmul_precision("high")
import pytorch_lightning as pl
from yacs.config import CfgNode as CN
from pytorch_lightning.tuner.tuning import Tuner
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.strategies import DDPStrategy
from loguru import logger as loguru_logger

from utils.profiler import build_profiler
from default_config import get_config
from utils.misc import setup_gpus, get_rank_zero_only_logger
from src.lightning_slim import PL_SLiM
from src.datasets.slim_dataset import SLiM_Dataset

# get logger and set as rank zero only(means ony info from rank 1 gpu will be stated out)
loguru_logger = get_rank_zero_only_logger(loguru_logger)


def cfg_get(cfg_node: CN, key: str, default):
    return cfg_node[key] if key in cfg_node else default


def parse_wandb_tags(tags: str):
    if not tags:
        return None
    return [tag.strip() for tag in tags.split(",") if tag.strip()]


def parse_wandb_log_model(value):
    if isinstance(value, bool) or value == "all":
        return value
    if str(value).lower() == "true":
        return True
    if str(value).lower() == "false":
        return False
    return value


def config_to_dict(config: CN):
    return yaml.safe_load(config.dump())


def resolve_epoch_checkpoint_interval(config: CN):
    save_every = int(config.TRAINER.SAVE_EVERY_N_EPOCHS)
    if save_every >= 0:
        return save_every

    save_every = max(1, int(config.TRAINER.MAX_EPOCHS / 10))
    config.TRAINER.SAVE_EVERY_N_EPOCHS = save_every
    loguru_logger.info(
        f"Auto epoch checkpoint interval: every {save_every} epochs "
        f"({config.TRAINER.MAX_EPOCHS} max epochs / 10)"
    )
    return save_every


def save_experiment_metadata(exp_dir: Path, config: CN, args, n_gpu_available: int):
    """Save reproducible experiment records next to checkpoints."""
    if int(os.environ.get("LOCAL_RANK", "0")) != 0:
        return

    paper_dir = exp_dir / "paper_logs"
    paper_dir.mkdir(parents=True, exist_ok=True)

    command = " ".join([sys.executable, *sys.argv])
    (paper_dir / "command.txt").write_text(command + "\n", encoding="utf-8")
    (paper_dir / "config.yaml").write_text(config.dump(), encoding="utf-8")
    (paper_dir / "config_pretty.txt").write_text(
        pprint.pformat(config) + "\n", encoding="utf-8"
    )
    env_lines = [
        f"python: {sys.version.split()[0]}",
        f"torch: {torch.__version__}",
        f"pytorch_lightning: {pl.__version__}",
        f"cuda_available: {torch.cuda.is_available()}",
        f"cuda_visible_devices: {os.environ.get('CUDA_VISIBLE_DEVICES', '')}",
        f"n_gpu_available: {n_gpu_available}",
        f"config_name: {args.config_name}",
        f"task_name: {args.task_name or config.LOGGER.LOGGER_NAME}",
        f"run_name: {args.run_name or ''}",
        f"max_epochs: {config.TRAINER.MAX_EPOCHS}",
        f"limit_train_batches: {config.TRAINER.LIMIT_TRAIN_BATCHES}",
        f"limit_val_batches: {config.TRAINER.LIMIT_VAL_BATCHES}",
        f"num_sanity_val_steps: {config.TRAINER.NUM_SANITY_VAL_STEPS}",
        f"save_every_n_epochs: {config.TRAINER.SAVE_EVERY_N_EPOCHS}",
        f"use_wandb: {config.LOGGER.USE_WANDB}",
        f"wandb_project: {config.LOGGER.WANDB_PROJECT}",
        f"wandb_entity: {config.LOGGER.WANDB_ENTITY}",
        f"wandb_mode: {config.LOGGER.WANDB_MODE}",
        f"wandb_log_model: {config.LOGGER.WANDB_LOG_MODEL}",
        f"resume_ckpt_path: {args.resume_ckpt_path or ''}",
    ]
    (paper_dir / "environment.txt").write_text(
        "\n".join(env_lines) + "\n", encoding="utf-8"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help='Comma-separated GPU ids. Use "0," for physical GPU 0; "1" means one visible GPU.',
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed.",
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default="outdoor_train",
        help="Config name. [outdoor_train, indoor_train, remote_optical_pairs_train, remote_optical_single_train, remote_multimodal_pairs_train]",
    )
    parser.add_argument("--max_epochs", type=int, default=None)
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Override config BATCH_SIZE and LOADER.BATCH_SIZE.",
    )
    parser.add_argument(
        "--accumulate_grad_batches",
        type=int,
        default=None,
        help="Override gradient accumulation steps.",
    )
    parser.add_argument("--limit_train_batches", type=float, default=None)
    parser.add_argument("--limit_val_batches", type=float, default=None)
    parser.add_argument("--num_sanity_val_steps", type=int, default=None)
    parser.add_argument(
        "--save_every_n_epochs",
        type=int,
        default=None,
        help="Save an epoch checkpoint every N epochs. Set -1 for max_epochs/10, 0 to disable.",
    )
    parser.add_argument(
        "--task_name",
        type=str,
        default=None,
        help="Custom task folder under logs/tb_logs. Defaults to config.LOGGER.LOGGER_NAME.",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help='Custom run folder under the task folder. Defaults to Lightning "version_N".',
    )
    parser.add_argument(
        "--resume_ckpt_path",
        type=str,
        default=None,
        help="Resume training state from a Lightning checkpoint, for example last.ckpt.",
    )
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        help="Enable Weights & Biases logging.",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default=None,
        help="Weights & Biases project name.",
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default=None,
        help="Weights & Biases entity/team name.",
    )
    parser.add_argument(
        "--wandb_mode",
        type=str,
        default=None,
        choices=["online", "offline", "disabled"],
        help="Weights & Biases mode.",
    )
    parser.add_argument(
        "--wandb_log_model",
        type=str,
        default=None,
        choices=["all", "True", "False"],
        help='Weights & Biases checkpoint logging. Use "all" to upload every checkpoint.',
    )
    parser.add_argument(
        "--wandb_tags",
        type=str,
        default=None,
        help="Comma-separated Weights & Biases tags.",
    )
    args = parser.parse_args()
    
    seed = args.seed
    config_name = args.config_name
    
    # get configurations
    config: CN = get_config(config_name)
    if args.device is not None:
        config.DEVICE.GPU_IDX = args.device
    config.TRAINER.set_new_allowed(True)
    config.LOGGER.set_new_allowed(True)
    if args.batch_size is not None:
        config.BATCH_SIZE = args.batch_size
        config.LOADER.BATCH_SIZE = args.batch_size
    config.TRAINER.ACCUMULATE_GRAD_BATCHES = (
        args.accumulate_grad_batches
        if args.accumulate_grad_batches is not None
        else cfg_get(config.TRAINER, "ACCUMULATE_GRAD_BATCHES", 1)
    )
    config.TRAINER.MAX_EPOCHS = (
        args.max_epochs
        if args.max_epochs is not None
        else cfg_get(config.TRAINER, "MAX_EPOCHS", 27)
    )
    config.TRAINER.LIMIT_TRAIN_BATCHES = (
        args.limit_train_batches
        if args.limit_train_batches is not None
        else cfg_get(config.TRAINER, "LIMIT_TRAIN_BATCHES", 1.0)
    )
    config.TRAINER.LIMIT_VAL_BATCHES = (
        args.limit_val_batches
        if args.limit_val_batches is not None
        else cfg_get(config.TRAINER, "LIMIT_VAL_BATCHES", 1.0)
    )
    config.TRAINER.NUM_SANITY_VAL_STEPS = (
        args.num_sanity_val_steps
        if args.num_sanity_val_steps is not None
        else cfg_get(config.TRAINER, "NUM_SANITY_VAL_STEPS", 2)
    )
    config.TRAINER.SAVE_EVERY_N_EPOCHS = (
        args.save_every_n_epochs
        if args.save_every_n_epochs is not None
        else cfg_get(config.TRAINER, "SAVE_EVERY_N_EPOCHS", -1)
    )
    config.LOGGER.USE_WANDB = bool(args.use_wandb or cfg_get(config.LOGGER, "USE_WANDB", False))
    config.LOGGER.WANDB_PROJECT = (
        args.wandb_project
        if args.wandb_project is not None
        else cfg_get(config.LOGGER, "WANDB_PROJECT", config.LOGGER.LOGGER_NAME)
    )
    config.LOGGER.WANDB_ENTITY = (
        args.wandb_entity
        if args.wandb_entity is not None
        else cfg_get(config.LOGGER, "WANDB_ENTITY", None)
    )
    config.LOGGER.WANDB_MODE = (
        args.wandb_mode
        if args.wandb_mode is not None
        else cfg_get(config.LOGGER, "WANDB_MODE", "online")
    )
    config.LOGGER.WANDB_LOG_MODEL = parse_wandb_log_model(
        args.wandb_log_model
        if args.wandb_log_model is not None
        else cfg_get(config.LOGGER, "WANDB_LOG_MODEL", "all")
    )
    config.LOGGER.WANDB_TAGS = (
        args.wandb_tags
        if args.wandb_tags is not None
        else cfg_get(config.LOGGER, "WANDB_TAGS", "")
    )
    
    # set seed
    if seed is None:
        import os
        import random

        seed = os.environ.get("GLOBAL_SEED")
        if seed:
            seed = int(seed)
        else:
            # set a random number with current time as random seed
            random.seed(a=None)
            seed = random.randint(0, 4294967295)
            os.environ["GLOBAL_SEED"] = str(seed)

    config.GLOBAL_SEED = seed
    pl.seed_everything(config.GLOBAL_SEED)
    torch.cuda.manual_seed_all(config.GLOBAL_SEED)

    # setup exact gpus available and set CUDA_VISIBLE_DEVICES variable
    n_gpu_available = (
        setup_gpus(config.DEVICE.GPU_IDX) if config.DEVICE.ENABLE_GPU else 0
    )
    config.DEVICE.ENABLE_DDP = n_gpu_available > 1 and config.DEVICE.ENABLE_DDP
    config.TRAINER.WORLD_SIZE = n_gpu_available * config.DEVICE.NUM_NODES
    config.TRAINER.TRUE_BATCH_SIZE = (
        config.TRAINER.WORLD_SIZE
        * config.BATCH_SIZE
        * config.TRAINER.ACCUMULATE_GRAD_BATCHES
    )
    config.TRAINER.SCALING = (
        config.TRAINER.TRUE_BATCH_SIZE / config.TRAINER.CANONICAL_BS
    )
    config.TRAINER.TRUE_LR = config.TRAINER.CANONICAL_LR * config.TRAINER.SCALING
    resolve_epoch_checkpoint_interval(config)
    loguru_logger.info("Config: \n" + pprint.pformat(config))

    # Profiler
    profiler = build_profiler(config.PROFILER.PROFILER_NAME)

    # Lightning module
    model = PL_SLiM(
        config=config, pretrained_ckpt=config.PRETRAINED_PATH, profiler=profiler
    )
    loguru_logger.info("Lightning Module initialized!")

    # Lightning data
    data_module = SLiM_Dataset(config=config)
    loguru_logger.info("Data Module initialized!")

    # Loggers
    task_name = args.task_name or config.LOGGER.LOGGER_NAME
    run_version = args.run_name if args.run_name else None
    tb_logger = TensorBoardLogger(
        save_dir="logs/tb_logs",
        name=task_name,
        version=run_version,
        default_hp_metric=False,
    )
    exp_dir = Path(tb_logger.log_dir)
    save_experiment_metadata(exp_dir, config, args, n_gpu_available)
    csv_logger = CSVLogger(save_dir=str(exp_dir / "paper_logs"), name="metrics", version="")
    loggers = [tb_logger, csv_logger]
    if config.LOGGER.USE_WANDB:
        try:
            from pytorch_lightning.loggers import WandbLogger
        except ImportError as exc:
            raise ImportError(
                "Weights & Biases logging was enabled, but wandb is not installed. "
                "Install it with `pip install wandb`, or run without `--use_wandb`."
            ) from exc

        wandb_logger = WandbLogger(
            project=config.LOGGER.WANDB_PROJECT,
            entity=config.LOGGER.WANDB_ENTITY,
            name=args.run_name or config.LOGGER.EXP_NAME,
            save_dir=str(exp_dir / "wandb"),
            log_model=config.LOGGER.WANDB_LOG_MODEL,
            mode=config.LOGGER.WANDB_MODE,
            tags=parse_wandb_tags(config.LOGGER.WANDB_TAGS),
        )
        wandb_logger.log_hyperparams(config_to_dict(config))
        loggers.append(wandb_logger)
    ckpt_dir = exp_dir / "checkpoints"
    is_remote_train = (
        str(config.DATASET.TRAINVAL_DATA_SOURCE).lower() == "remotesensing"
    )

    # Callbacks
    if config.TRAINER.LIMIT_VAL_BATCHES == 0:
        best_ckpt_callback = ModelCheckpoint(
            verbose=True,
            save_top_k=0,
            save_last=True,
            dirpath=str(ckpt_dir),
            filename="{epoch}",
        )
    elif is_remote_train:
        best_ckpt_callback = ModelCheckpoint(
            monitor="remote_inlier@5",
            verbose=True,
            save_top_k=1,
            mode="max",
            save_last=True,
            dirpath=str(ckpt_dir),
            filename="best-{epoch}-{remote_inlier@5:.3f}-{remote_median_error:.2f}",
        )
    else:
        best_ckpt_callback = ModelCheckpoint(
            monitor="auc@5",
            verbose=True,
            save_top_k=1,
            mode="max",
            save_last=True,
            dirpath=str(ckpt_dir),
            filename="best-{epoch}-{auc@5:.3f}-{auc@10:.3f}-{auc@20:.3f}",
        )
    callbacks = [best_ckpt_callback]
    if config.TRAINER.SAVE_EVERY_N_EPOCHS > 0:
        epoch_ckpt_callback = ModelCheckpoint(
            verbose=True,
            save_top_k=-1,
            save_last=False,
            every_n_epochs=config.TRAINER.SAVE_EVERY_N_EPOCHS,
            dirpath=str(ckpt_dir / "epoch_checkpoints"),
            filename="epoch-{epoch:03d}",
            auto_insert_metric_name=False,
        )
        callbacks.append(epoch_ckpt_callback)
    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks.append(lr_monitor)

    # Torch Lightning Trainer
    trainer = pl.Trainer(
        accelerator="gpu" if config.DEVICE.ENABLE_GPU else "cpu",
        strategy=DDPStrategy(find_unused_parameters=True)
        if config.DEVICE.ENABLE_DDP
        else "auto",
        devices=n_gpu_available,
        num_nodes=config.DEVICE.NUM_NODES,
        logger=loggers,
        callbacks=callbacks,
        gradient_clip_val=config.TRAINER.GRADIENT_CLIPPING,
        sync_batchnorm=(config.TRAINER.WORLD_SIZE > 1),
        profiler=profiler,
        accumulate_grad_batches=config.TRAINER.ACCUMULATE_GRAD_BATCHES,
        log_every_n_steps=max(int(50 / config.TRAINER.ACCUMULATE_GRAD_BATCHES), 1),
        max_epochs=config.TRAINER.MAX_EPOCHS,
        limit_train_batches=config.TRAINER.LIMIT_TRAIN_BATCHES,
        limit_val_batches=config.TRAINER.LIMIT_VAL_BATCHES,
        num_sanity_val_steps=config.TRAINER.NUM_SANITY_VAL_STEPS,
        check_val_every_n_epoch=1,
    )
    loguru_logger.info("Trainer Initialized!")

    if config.TRAINER.FIND_LR:
        # Finding best LR with linear progression
        temp = config.TRAINER.WARMUP_TYPE
        model.num_devices = n_gpu_available
        tuner = Tuner(trainer)
        lr_finder = tuner.lr_find(model=model, datamodule=data_module, num_training=300)
        print(
            f"Best LR found by LR finder with linear progression :{lr_finder.suggestion()}"
        )
        # Setting best LR
        config.TRAINER.TRUE_LR = lr_finder.suggestion()
        model.lr = lr_finder.suggestion()
        config.TRAINER.WARMUP_TYPE = temp

    # Training
    model.num_devices = trainer.num_devices
    loguru_logger.info("Start training!")
    trainer.fit(model, datamodule=data_module, ckpt_path=args.resume_ckpt_path)


if __name__ == "__main__":
    main()
