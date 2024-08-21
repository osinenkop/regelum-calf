# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import logging

from omegaconf import DictConfig

import rehydra
from rehydra.core.rehydra_config import RehydraConfig

log = logging.getLogger(__name__)


@rehydra.main(version_base=None, config_path=".", config_name="config")
def my_app(cfg: DictConfig) -> None:
    log.info(f"Output_dir={RehydraConfig.get().runtime.output_dir}")
    log.info(f"cfg.foo={cfg.foo}")


if __name__ == "__main__":
    my_app()
