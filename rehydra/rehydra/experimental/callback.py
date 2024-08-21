# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import logging
from typing import Any

from omegaconf import DictConfig

from rehydra.core.utils import JobReturn
from rehydra.types import TaskFunction

logger = logging.getLogger(__name__)


class Callback:
    def on_run_start(self, config: DictConfig, **kwargs: Any) -> None:
        """
        Called in RUN mode before job/application code starts. `config` is composed with overrides.
        Some `rehydra.runtime` configs are not populated yet.
        See rehydra.core.utils.run_job for more info.
        """
        ...

    def on_run_end(self, config: DictConfig, **kwargs: Any) -> None:
        """
        Called in RUN mode after job/application code returns.
        """
        ...

    def on_multirun_start(self, config: DictConfig, **kwargs: Any) -> None:
        """
        Called in MULTIRUN mode before any job starts.
        When using a launcher, this will be executed on local machine before any Sweeper/Launcher is initialized.
        """
        ...

    def on_multirun_end(self, config: DictConfig, **kwargs: Any) -> None:
        """
        Called in MULTIRUN mode after all jobs returns.
        When using a launcher, this will be executed on local machine.
        """
        ...

    def on_job_start(
        self, config: DictConfig, *, task_function: TaskFunction, **kwargs: Any
    ) -> None:
        """
        Called in both RUN and MULTIRUN modes, once for each Rehydra job (before running application code).
        This is called from within `rehydra.core.utils.run_job`. In the case of remote launching, this will be executed
        on the remote server along with your application code. The `task_function` argument is the function
        decorated with `@rehydra.main`.
        """
        ...

    def on_job_end(
        self, config: DictConfig, job_return: JobReturn, **kwargs: Any
    ) -> None:
        """
        Called in both RUN and MULTIRUN modes, once for each Rehydra job (after running
        application code).
        This is called from within `rehydra.core.utils.run_job`. In the case of remote launching, this will be executed
        on the remote server after your application code.

        `job_return` contains info that could be useful for logging or post-processing.
        See rehydra.core.utils.JobReturn for more.
        """
        ...
