# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from typing import List, Optional

from rehydra.plugins.sweeper import Sweeper
from rehydra.types import RehydraContext, TaskFunction
from omegaconf import DictConfig

from .config import AxConfig


class AxSweeper(Sweeper):
    """Class to interface with the Ax Platform"""

    def __init__(self, ax_config: AxConfig, max_batch_size: Optional[int]):
        from ._core import CoreAxSweeper

        self.sweeper = CoreAxSweeper(ax_config, max_batch_size)

    def setup(
        self,
        *,
        rehydra_context: RehydraContext,
        task_function: TaskFunction,
        config: DictConfig,
    ) -> None:
        return self.sweeper.setup(
            rehydra_context=rehydra_context, task_function=task_function, config=config
        )

    def sweep(self, arguments: List[str]) -> None:
        return self.sweeper.sweep(arguments)
