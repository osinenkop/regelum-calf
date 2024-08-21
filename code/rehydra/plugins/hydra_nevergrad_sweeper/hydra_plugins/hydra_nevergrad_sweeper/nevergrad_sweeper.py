# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from typing import List, Optional

from rehydra import TaskFunction
from rehydra.plugins.sweeper import Sweeper
from rehydra.types import RehydraContext
from omegaconf import DictConfig

from .config import OptimConf


class NevergradSweeper(Sweeper):
    """Class to interface with Nevergrad"""

    def __init__(self, optim: OptimConf, parametrization: Optional[DictConfig]):
        from ._impl import NevergradSweeperImpl

        self.sweeper = NevergradSweeperImpl(optim, parametrization)

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
