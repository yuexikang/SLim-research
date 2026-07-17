from configs.cfg_outdoor_train import get_cfg_defaults as get_outdoor_cfg  # 复用原 outdoor/MegaDepth 训练配置作为基础模板


def get_cfg_defaults():  # 返回“遥感多模态成对影像”训练配置
    cfg = get_outdoor_cfg()  # 先加载原项目默认 outdoor 训练配置
    cfg.defrost()  # 解冻 yacs 配置，允许下面覆盖字段
    cfg.set_new_allowed(True)  # 允许在根配置下新增遥感专用字段
    cfg.DATASET.set_new_allowed(True)  # 允许在 DATASET 节点下新增遥感专用字段
    cfg.LOGGER.set_new_allowed(True)  # 允许在 LOGGER 节点下新增 W&B 相关字段

    cfg.MODE = "remote_multimodal_pairs_train"  # 当前配置名称，用来标识训练模式
    cfg.PRETRAINED_PATH = "ckpt/megadepth_19epochs.ckpt"  # 从 MegaDepth 预训练权重开始微调
    cfg.BATCH_SIZE = 4  # 每张 GPU 每个 step 输入 4 对多模态遥感影像
    cfg.IMAGE_SIZE = 512  # 遥感影像统一 resize 到 512x512 后训练
    cfg.AMP = True  # 开启混合精度训练，节省显存并提升速度

    cfg.DEVICE.GPU_IDX = "0,"  # 默认使用物理 0 号 GPU；注意带逗号表示指定 GPU id
    cfg.DEVICE.ENABLE_DDP = False  # 默认不开多卡 DDP；多卡时可改 True 或由 train.py 自动处理

    cfg.DATASET.DATA_SOURCE = "RemoteSensing"  # 数据集类型标记为遥感数据
    cfg.DATASET.TRAINVAL_DATA_SOURCE = "RemoteSensing"  # 训练/验证都走遥感数据加载器
    cfg.DATASET.REMOTE_TRAIN_MANIFEST = "data/remote_archive/manifests/train_multimodal_pairs.jsonl"  # 训练多模态成对影像索引
    cfg.DATASET.REMOTE_VAL_MANIFEST = "data/remote_archive/manifests/val_multimodal_pairs.jsonl"  # 验证多模态成对影像索引
    cfg.DATASET.REMOTE_IMAGE_SIZE = cfg.IMAGE_SIZE  # 数据加载器使用的 resize 尺寸，与全局 IMAGE_SIZE 保持一致
    cfg.DATASET.REMOTE_MAX_TRAIN_SAMPLES = 0  # 训练集最多读取多少条；当前是调试量，0 表示不截断 manifest
    cfg.DATASET.REMOTE_MAX_VAL_SAMPLES = 0  # 验证集最多读取多少条；当前是调试量，0 表示不截断 manifest
    cfg.DATASET.REMOTE_VAL_SPLIT_RATIO = 0.2  # 没有填写验证 manifest 时，从训练 manifest 中按这个比例拆出验证集
    cfg.DATASET.REMOTE_HOMOGRAPHY_DIFFICULTY = 0.25  # 在线合成 homography 的扰动强度，越大变换越难
    cfg.DATASET.REMOTE_LEFT_IDENTITY = True  # image0 保持原图，主要扰动 image1，便于稳定训练
    cfg.DATASET.REMOTE_AUG_VARIANTS = ["translation", "scale", "yaw", "pitch", "roll"]  # 在线生成的几何变换类型；每条原始样本会按列表展开，不保存到本地

    cfg.SAMPLER.N_SAMPLES_PER_SUBSET = 0  # 每个 epoch 从该数据子集采样多少对；0 表示自动使用当前训练集全部样本
    cfg.SAMPLER.SUBSET_REPLACEMENT = False  # 采样时优先不放回；样本不够时 sampler 会补齐
    cfg.SAMPLER.REPEAT = 1  # 每个 epoch 内重复采样结果的次数
    cfg.LOADER.BATCH_SIZE = cfg.BATCH_SIZE  # DataLoader 使用的 batch size，与全局 BATCH_SIZE 同步
    cfg.LOADER.NUM_WORKERS = 4  # DataLoader 进程数；正式训练可调到 4/8/16 提升吞吐
    cfg.LOADER.PIN_MEMORY = True  # 固定内存，加快 CPU 到 GPU 的数据拷贝

    cfg.MODEL.COARSE_SCALE = cfg.MODEL.BACKBONE.PATCH_SIZE * 2 ** cfg.COARSE_SCALE_IDX  # 粗匹配特征相对原图的下采样倍率
    cfg.MODEL.MAX_COARSE_MATCHES = 1024  # 每个 batch 最多保留的粗匹配数量
    cfg.MODEL.MAX_FINE_MATCHES = 1024  # 每个 batch 最多进入精细匹配/细化的匹配数量

    cfg.TRAINER.CANONICAL_BS = 8  # 标称 batch size，用于按实际 batch 自动缩放学习率
    cfg.TRAINER.CANONICAL_LR = 1e-4  # 标称学习率，实际 TRUE_LR 会按 batch size 缩放
    cfg.TRAINER.WARMUP_STEP = 0  # warmup 步数；0 表示不做 warmup
    cfg.TRAINER.N_VAL_PAIRS_TO_PLOT = 1  # 原验证可视化对数；遥感验证当前主要看数值指标
    cfg.TRAINER.ENABLE_PLOTTING = False  # 关闭验证匹配图绘制，减少训练开销
    cfg.TRAINER.MAX_EPOCHS = 27  # 默认训练 27 个 epoch，可被 train.py 的 --max_epochs 临时覆盖
    cfg.TRAINER.LIMIT_TRAIN_BATCHES = 1.0  # 每个 epoch 使用全部训练 batch；小于 1 表示按比例截断
    cfg.TRAINER.LIMIT_VAL_BATCHES = 1.0  # 每次验证使用全部验证 batch；设为 0 可关闭验证
    cfg.TRAINER.NUM_SANITY_VAL_STEPS = 0  # 训练前不额外跑 sanity validation，避免启动时多耗时间
    cfg.TRAINER.SAVE_EVERY_N_EPOCHS = -1  # 阶段权重按 epoch 结束保存；-1 表示每 1/10 总 epoch 保存一次，0 关闭，正数表示固定 epoch 间隔
    cfg.TRAINER.ACCUMULATE_GRAD_BATCHES = 1  # 梯度累积步数；>1 时每 accumulate_grad_batches 个 step 才更新一次梯度，等效于增大 batch size

    cfg.LOGGER.LOGGER_NAME = "remote_multimodal_pairs"  # 默认日志任务名，可被 train.py 的 --task_name 覆盖
    cfg.LOGGER.USE_WANDB = False  # 默认不启用 W&B；运行时加 --use_wandb 即可打开
    cfg.LOGGER.WANDB_PROJECT = "slim_remote_sensing"  # W&B 项目名，可被 --wandb_project 覆盖
    cfg.LOGGER.WANDB_ENTITY = None  # W&B 团队/用户名；None 表示使用当前登录账号默认 entity
    cfg.LOGGER.WANDB_MODE = "online"  # W&B 模式：online 正常上传，offline 本地缓存，disabled 禁用
    cfg.LOGGER.WANDB_LOG_MODEL = "all"  # 记录 checkpoint 到 W&B；all 表示记录所有 ModelCheckpoint 产物
    cfg.LOGGER.WANDB_TAGS = "remote,multimodal,pairs"  # W&B 标签，便于筛选实验
    return cfg  # 返回最终配置
