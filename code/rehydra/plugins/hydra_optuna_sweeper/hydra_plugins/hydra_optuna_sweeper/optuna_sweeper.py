# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from typing import Any, List, Optional

from rehydra.plugins.sweeper import Sweeper
from rehydra.types import RehydraContext, TaskFunction
from omegaconf import DictConfig

from .config import SamplerConfig


class OptunaSweeper(Sweeper):
    """Class to interface with Optuna"""

    def __init__(
        self,
        sampler: SamplerConfig,
        direction: Any,
        storage: Optional[Any],
        study_name: Optional[str],
        n_trials: int,
        n_jobs: int,
        max_failure_rate: float,
        search_space: Optional[DictConfig],
        custom_search_space: Optional[str],
        params: Optional[DictConfig],
    ) -> None:
        from ._impl import OptunaSweeperImpl

        self.sweeper = OptunaSweeperImpl(
            sampler,
            direction,
            storage,
            study_name,
            n_trials,
            n_jobs,
            max_failure_rate,
            search_space,
            custom_search_space,
            params,
        )

    def setup(
        self,
        *,
        rehydra_context: RehydraContext,
        task_function: TaskFunction,
        config: DictConfig,
    ) -> None:
        self.sweeper.setup(
            rehydra_context=rehydra_context, task_function=task_function, config=config
        )

    def sweep(self, arguments: List[str]) -> None:
        return self.sweeper.sweep(arguments)
