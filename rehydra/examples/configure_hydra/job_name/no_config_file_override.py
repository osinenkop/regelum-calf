# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from omegaconf import DictConfig

import rehydra
from rehydra.core.rehydra_config import RehydraConfig


@rehydra.main(version_base=None)
def experiment(_cfg: DictConfig) -> None:
    print(RehydraConfig.get().job.name)


if __name__ == "__main__":
    experiment()
