# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import logging
import time

import rehydra
from omegaconf import DictConfig

log = logging.getLogger(__name__)


@rehydra.main(version_base=None, config_name="config", config_path=".")
def my_app(cfg: DictConfig) -> None:
    log.info(f"Executing task {cfg.task}")
    time.sleep(1)


if __name__ == "__main__":
    my_app()
