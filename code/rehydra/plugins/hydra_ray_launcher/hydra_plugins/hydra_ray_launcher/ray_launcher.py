# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from typing import Optional, Sequence

from rehydra.core.utils import JobReturn
from rehydra.plugins.launcher import Launcher
from rehydra.types import RehydraContext, TaskFunction
from omegaconf import DictConfig


class RayLauncher(Launcher):
    def __init__(self, ray: DictConfig) -> None:
        self.ray_cfg = ray
        self.rehydra_context: Optional[RehydraContext] = None
        self.task_function: Optional[TaskFunction] = None
        self.config: Optional[DictConfig] = None

    def setup(
        self,
        *,
        rehydra_context: RehydraContext,
        task_function: TaskFunction,
        config: DictConfig,
    ) -> None:
        self.config = config
        self.rehydra_context = rehydra_context
        self.task_function = task_function

    def launch(
        self, job_overrides: Sequence[Sequence[str]], initial_job_idx: int
    ) -> Sequence[JobReturn]:
        from . import _core

        return _core.launch(
            launcher=self, job_overrides=job_overrides, initial_job_idx=initial_job_idx
        )
