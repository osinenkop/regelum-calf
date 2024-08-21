# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from omegaconf import OmegaConf

from rehydra import compose, initialize

if __name__ == "__main__":
    # initialize the Rehydra subsystem.
    # This is needed for apps that cannot have a standard @rehydra.main() entry point
    initialize(version_base=None, config_path="conf")
    cfg = compose("config.yaml", overrides=["db=mysql", "db.user=${oc.env:USER}"])
    print(OmegaConf.to_yaml(cfg, resolve=True))
