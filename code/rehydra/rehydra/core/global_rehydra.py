# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from typing import Any, Optional

from rehydra._internal.rehydra import Rehydra
from rehydra.core.config_loader import ConfigLoader
from rehydra.core.singleton import Singleton


class GlobalRehydra(metaclass=Singleton):
    def __init__(self) -> None:
        self.rehydra: Optional[Rehydra] = None

    def initialize(self, rehydra: "Rehydra") -> None:
        assert isinstance(rehydra, Rehydra), f"Unexpected Rehydra type : {type(rehydra)}"
        if self.is_initialized():
            raise ValueError(
                "GlobalRehydra is already initialized, call GlobalRehydra.instance().clear() if you want to re-initialize"
            )
        self.rehydra = rehydra

    def config_loader(self) -> "ConfigLoader":
        assert self.rehydra is not None
        return self.rehydra.config_loader

    def is_initialized(self) -> bool:
        return self.rehydra is not None

    def clear(self) -> None:
        self.rehydra = None

    @staticmethod
    def instance(*args: Any, **kwargs: Any) -> "GlobalRehydra":
        return Singleton.instance(GlobalRehydra, *args, **kwargs)  # type: ignore

    @staticmethod
    def set_instance(instance: "GlobalRehydra") -> None:
        assert isinstance(instance, GlobalRehydra)
        Singleton._instances[GlobalRehydra] = instance  # type: ignore
