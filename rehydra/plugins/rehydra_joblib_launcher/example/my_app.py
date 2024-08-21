# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import logging
import os
import time

import rehydra
from omegaconf import DictConfig

log = logging.getLogger(__name__)


@rehydra.main(version_base=None, config_name="config")
def my_app(cfg: DictConfig) -> None:
    log.info(f"Process ID {os.getpid()} executing task {cfg.task} ...")

    time.sleep(1)


if __name__ == "__main__":
    my_app()
