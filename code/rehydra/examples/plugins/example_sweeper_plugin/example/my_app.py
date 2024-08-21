# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import rehydra
from omegaconf import OmegaConf, DictConfig


@rehydra.main(version_base=None, config_path="conf", config_name="config")
def my_app(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))


if __name__ == "__main__":
    my_app()
