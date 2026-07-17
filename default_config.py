def get_config(config_name: str):
    """
    Get the default config.
    """
    if config_name == "outdoor_train":
        from configs.cfg_outdoor_train import get_cfg_defaults
    elif config_name == "outdoor_test":
        from configs.cfg_outdoor_test import get_cfg_defaults
    elif config_name == "indoor_train":
        from configs.cfg_indoor_train import get_cfg_defaults
    elif config_name == "indoor_test":
        from configs.cfg_indoor_test import get_cfg_defaults
    elif config_name == "remote_optical_pairs_train":
        from configs.cfg_remote_optical_pairs_train import get_cfg_defaults
    elif config_name == "remote_optical_single_train":
        from configs.cfg_remote_optical_single_train import get_cfg_defaults
    elif config_name == "remote_multimodal_pairs_train":
        from configs.cfg_remote_multimodal_pairs_train import get_cfg_defaults
    else:
        raise ValueError(f"Unknown config name: {config_name}")
    return get_cfg_defaults()
