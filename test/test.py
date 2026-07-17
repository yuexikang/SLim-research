# 作用：运行 SLiM 项目的原生数据集测试流程。
# 测试什么数据集：
#   1. --config_name outdoor_test 测 MegaDepth 测试集，默认读取 data/megadepth/test 和
#      data/megadepth/index/scene_info_val_1500，输出 epipolar/pose AUC 指标。
#   2. --config_name indoor_test 测 ScanNet 测试集，默认读取 data/scannet/test 和对应 index。
# 怎么测试：
#   MegaDepth 示例：
#   MPLCONFIGDIR=/tmp/matplotlib python test/test.py --config_name outdoor_test \
#       --ckpt_path ckpt/megadepth_19epochs.ckpt --device "2," --amp
# 注意：--device "2," 表示只用物理 GPU 2；--device 2 会被解释为使用前 2 张 GPU。
# 加速建议：
#   1. 多卡测试：--device "2,3" 会用物理 GPU 2 和 3，并自动启用 DDP。
#   2. 数据读取较慢时：加 --num_workers 8 或 12。
#   3. 先试跑少量数据：加 --limit_test_batches 20。
import argparse
import math
import pprint
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from yacs.config import CfgNode as CN
import pytorch_lightning as pl
from loguru import logger as loguru_logger

from utils.profiler import build_profiler
from default_config import get_config
from utils.misc import setup_gpus, get_rank_zero_only_logger
from src.lightning_slim import PL_SLiM
from src.datasets.slim_dataset import SLiM_Dataset

# get logger and set as rank zero only(means ony info from rank 1 gpu will be stated out)
loguru_logger = get_rank_zero_only_logger(loguru_logger)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="ckpt/megadepth_19epochs.ckpt",
        help="Path to checkpoint file.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="Comma-separated list of GPU devices to use.",
    )
    parser.add_argument(
        "--thr",
        type=float,
        default=None,
        help="Threshold used in correlation filtering.",
    )
    parser.add_argument(
        "--refine_iters",
        type=int,
        default=4,
        help="Iterations number for refinement.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed.",
    )
    parser.add_argument(
        "--optimized_ds",
        action="store_true",
        help="Partial dual softmax.",
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Auto mixed precision.",
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default="outdoor_test",
        help="Config name. [outdoor_test, indoor_test]",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="Override DataLoader workers. Try 8 or 12 if CPU/IO is the bottleneck.",
    )
    parser.add_argument(
        "--limit_test_batches",
        type=int,
        default=None,
        help="Run only the first N test batches. Useful for quick checks.",
    )
    parser.add_argument(
        "--matmul_precision",
        type=str,
        default=None,
        choices=["highest", "high", "medium"],
        help="Set torch float32 matmul precision for Tensor Cores. medium is fastest.",
    )
    args = parser.parse_args()

    ckpt_path = args.ckpt_path
    device = args.device
    thr = args.thr
    refine_iters = args.refine_iters
    seed = args.seed
    optimized_ds = args.optimized_ds
    amp = args.amp
    config_name = args.config_name
    num_workers = args.num_workers
    limit_test_batches = args.limit_test_batches
    matmul_precision = args.matmul_precision

    # get configurations
    config: CN = get_config(config_name)
    if matmul_precision is not None:
        torch.set_float32_matmul_precision(matmul_precision)
    
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
    n_gpu_available = setup_gpus(device) if config.DEVICE.ENABLE_GPU else 0
    config.DEVICE.ENABLE_DDP = n_gpu_available > 1
    config.TRAINER.WORLD_SIZE = n_gpu_available * config.DEVICE.NUM_NODES
    config.TRAINER.TRUE_BATCH_SIZE = (
        config.TRAINER.WORLD_SIZE * config.LOADER.BATCH_SIZE
    )
    config.TRAINER.SCALING = (
        config.TRAINER.TRUE_BATCH_SIZE / config.TRAINER.CANONICAL_BS
    )
    config.TRAINER.TRUE_LR = config.TRAINER.CANONICAL_LR * config.TRAINER.SCALING
    config.TRAINER.WARMUP_STEP = math.floor(
        config.TRAINER.WARMUP_STEP / config.TRAINER.SCALING
    )
    # Set threshold if provided
    if thr is not None:
        config.MODEL.COARSE_THRES = config.COARSE_THRES = thr
        config.MODEL.FINE_THRES = config.FINE_THRES = thr
    if num_workers is not None:
        config.LOADER.NUM_WORKERS = num_workers
    config.MODEL.REFINE_ITERS = config.REFINE_ITERS = refine_iters
    config.DUMP_DIR = {
        "outdoor_train": None,
        "outdoor_test": f"dump/slim_outdoor_I{config.REFINE_ITERS}",
        "indoor_train": None,
        "indoor_test": f"dump/slim_indoor_I{config.REFINE_ITERS}",
    }[config.MODE]
    config.MODEL.OPTIMIZED_DUAL_SOFTMAX = config.OPTIMIZED_DUAL_SOFTMAX = optimized_ds
    config.AMP = amp if amp is not None else config.AMP
    loguru_logger.info("Config: \n" + pprint.pformat(config))

    # Profiler
    profiler = build_profiler("inference")

    # Lightning module
    model = PL_SLiM(
        config=config,
        pretrained_ckpt=ckpt_path,
        profiler=profiler,
        dump_dir=config.DUMP_DIR,
    )
    loguru_logger.info("Lightning Module initialized!")

    # Lightning data
    data_module = SLiM_Dataset(config=config)
    loguru_logger.info("Data Module initialized!")

    # Torch Lightning Trainer
    trainer = pl.Trainer(
        accelerator="gpu" if config.DEVICE.ENABLE_GPU else "cpu",
        devices=n_gpu_available,
        num_nodes=config.DEVICE.NUM_NODES,
        profiler=profiler,
        logger=False,
        limit_test_batches=limit_test_batches,
        strategy="ddp" if config.DEVICE.ENABLE_DDP else "auto",
    )
    loguru_logger.info("Trainer Initialized!")

    # Testing
    loguru_logger.info("Start testing!")
    trainer.test(model, datamodule=data_module)


if __name__ == "__main__":
    main()
