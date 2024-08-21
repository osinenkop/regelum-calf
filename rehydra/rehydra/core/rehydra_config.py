# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from typing import Any, Optional

from omegaconf import DictConfig, OmegaConf

from rehydra.conf import RehydraConf
from rehydra.core.singleton import Singleton


class RehydraConfig(metaclass=Singleton):
    def __init__(self) -> None:
        self.cfg: Optional[RehydraConf] = None

    def set_config(self, cfg: DictConfig) -> None:
        assert cfg is not None
        OmegaConf.set_readonly(cfg.rehydra, True)
        rehydra_node_type = OmegaConf.get_type(cfg, "rehydra")
        assert rehydra_node_type is not None and issubclass(rehydra_node_type, RehydraConf)
        # THis is emulating a node that is hidden.
        # It's quiet a hack but it will be much better once
        # https://github.com/omry/omegaconf/issues/280 is done
        # The motivation is that this allows for interpolations from the rehydra node
        # into the user's config.
        self.cfg = OmegaConf.masked_copy(cfg, "rehydra")  # type: ignore
        self.cfg.rehydra._set_parent(cfg)  # type: ignore

    @staticmethod
    def get() -> RehydraConf:
        instance = RehydraConfig.instance()
        if instance.cfg is None:
            raise ValueError("RehydraConfig was not set")
        return instance.cfg.rehydra  # type: ignore

    @staticmethod
    def initialized() -> bool:
        instance = RehydraConfig.instance()
        return instance.cfg is not None

    @staticmethod
    def instance(*args: Any, **kwargs: Any) -> "RehydraConfig":
        return Singleton.instance(RehydraConfig, *args, **kwargs)  # type: ignore
