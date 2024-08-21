# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import logging
from typing import Any, Optional, Sequence

from rehydra.core.utils import JobReturn
from rehydra.plugins.launcher import Launcher
from rehydra.types import RehydraContext, TaskFunction
from omegaconf import DictConfig, OmegaConf

from .config import RQLauncherConf

log = logging.getLogger(__name__)


class RQLauncher(Launcher):
    def __init__(self, **params: Any) -> None:
        """RQ Launcher

        Launches jobs using RQ (Redis Queue). For details, refer to:
        https://python-rq.org
        """
        self.config: Optional[DictConfig] = None
        self.task_function: Optional[TaskFunction] = None
        self.rehydra_context: Optional[RehydraContext] = None

        self.rq = OmegaConf.structured(RQLauncherConf(**params))

    def setup(
        self,
        *,
        rehydra_context: RehydraContext,
        task_function: TaskFunction,
        config: DictConfig,
    ) -> None:
        self.config = config
        self.task_function = task_function
        self.rehydra_context = rehydra_context

    def launch(
        self, job_overrides: Sequence[Sequence[str]], initial_job_idx: int
    ) -> Sequence[JobReturn]:
        from . import _core

        return _core.launch(
            launcher=self, job_overrides=job_overrides, initial_job_idx=initial_job_idx
        )
