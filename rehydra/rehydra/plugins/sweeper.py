# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Sweeper plugin interface
"""
from abc import abstractmethod
from typing import Any, List, Sequence, Optional

from rehydra.types import TaskFunction
from omegaconf import DictConfig
from .launcher import Launcher

from .plugin import Plugin
from rehydra.types import RehydraContext


class Sweeper(Plugin):
    """
    An abstract sweeper interface
    Sweeper takes the command line arguments, generates a and launches jobs
    (where each job typically takes a different command line arguments)
    """

    rehydra_context: Optional[RehydraContext]
    config: Optional[DictConfig]
    launcher: Optional[Launcher]

    @abstractmethod
    def setup(
        self,
        *,
        rehydra_context: RehydraContext,
        task_function: TaskFunction,
        config: DictConfig,
    ) -> None:
        raise NotImplementedError()

    @abstractmethod
    def sweep(self, arguments: List[str]) -> Any:
        """
        Execute a sweep
        :param arguments: list of strings describing what this sweeper should do.
        exact structure is determine by the concrete Sweeper class.
        :return: the return objects of all thy launched jobs. structure depends on the Sweeper
        implementation.
        """
        ...

    def validate_batch_is_legal(self, batch: Sequence[Sequence[str]]) -> None:
        """
        Ensures that the given batch can be composed.
        This repeat work the launcher will do, but as the launcher may be performing this in a different
        process/machine it's important to do it here as well to detect failures early.
        """
        config_loader = (
            self.rehydra_context.config_loader
            if hasattr(self, "rehydra_context") and self.rehydra_context is not None
            else self.config_loader  # type: ignore
        )
        assert config_loader is not None

        assert self.config is not None
        for overrides in batch:
            config_loader.load_sweep_config(
                master_config=self.config, sweep_overrides=list(overrides)
            )
