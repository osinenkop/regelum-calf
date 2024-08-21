# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import rehydra

# Generated config dataclasses
from example.config.configen.samples.my_module import AdminConf, UserConf
from rehydra.core.config_store import ConfigStore
from omegaconf import DictConfig

# Underlying objects
from configen.samples.my_module import Admin, User

ConfigStore.instance().store(
    name="config_schema",
    node={
        "user": UserConf,
        "admin": AdminConf,
    },
)


@rehydra.main(version_base=None, config_path=".", config_name="config")
def my_app(cfg: DictConfig) -> None:
    user: User = rehydra.utils.instantiate(cfg.user)
    admin: Admin = rehydra.utils.instantiate(cfg.admin)
    print(user)
    print(admin)


if __name__ == "__main__":
    my_app()
