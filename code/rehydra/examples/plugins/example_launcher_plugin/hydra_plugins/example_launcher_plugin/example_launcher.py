# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from dataclasses import dataclass

import logging
from pathlib import Path
from typing import Optional, Sequence

from rehydra.types import RehydraContext
from rehydra.core.config_store import ConfigStore
from rehydra.core.singleton import Singleton
from rehydra.core.utils import (
    JobReturn,
    configure_log,
    filter_overrides,
    run_job,
    setup_globals,
)
from rehydra.plugins.launcher import Launcher
from rehydra.types import TaskFunction
from omegaconf import DictConfig, open_dict

# IMPORTANT:
# If your plugin imports any module that takes more than a fraction of a second to import,
# Import the module lazily (typically inside launch()).
# Installed plugins are imported during Rehydra initialization and plugins that are slow to import plugins will slow
# the startup of ALL rehydra applications.
# Another approach is to place heavy includes in a file prefixed by _, such as _core.py:
# Rehydra will not look for plugin in such files and will not import them during plugin discovery.


log = logging.getLogger(__name__)


@dataclass
class LauncherConfig:
    _target_: str = (
        "rehydra_plugins.example_launcher_plugin.example_launcher.ExampleLauncher"
    )
    foo: int = 10
    bar: str = "abcde"


ConfigStore.instance().store(
    group="rehydra/launcher", name="example", node=LauncherConfig
)


class ExampleLauncher(Launcher):
    def __init__(self, foo: str, bar: str) -> None:
        self.config: Optional[DictConfig] = None
        self.task_function: Optional[TaskFunction] = None
        self.rehydra_context: Optional[RehydraContext] = None

        # foo and var are coming from the the plugin's configuration
        self.foo = foo
        self.bar = bar

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
        """
        :param job_overrides: a List of List<String>, where each inner list is the arguments for one job run.
        :param initial_job_idx: Initial job idx in batch.
        :return: an array of return values from run_job with indexes corresponding to the input list indexes.
        """
        setup_globals()
        assert self.config is not None
        assert self.rehydra_context is not None
        assert self.task_function is not None

        configure_log(self.config.rehydra.rehydra_logging, self.config.rehydra.verbose)
        sweep_dir = Path(str(self.config.rehydra.sweep.dir))
        sweep_dir.mkdir(parents=True, exist_ok=True)
        log.info(
            f"Example Launcher(foo={self.foo}, bar={self.bar}) is launching {len(job_overrides)} jobs locally"
        )
        log.info(f"Sweep output dir : {sweep_dir}")
        runs = []

        for idx, overrides in enumerate(job_overrides):
            idx = initial_job_idx + idx
            lst = " ".join(filter_overrides(overrides))
            log.info(f"\t#{idx} : {lst}")
            sweep_config = self.rehydra_context.config_loader.load_sweep_config(
                self.config, list(overrides)
            )
            with open_dict(sweep_config):
                # This typically coming from the underlying scheduler (SLURM_JOB_ID for instance)
                # In that case, it will not be available here because we are still in the main process.
                # but instead should be populated remotely before calling the task_function.
                sweep_config.rehydra.job.id = f"job_id_for_{idx}"
                sweep_config.rehydra.job.num = idx

            # If your launcher is executing code in a different process, it is important to restore
            # the singleton state in the new process.
            # To do this, you will likely need to serialize the singleton state along with the other
            # parameters passed to the child process.

            # happening on this process (executing launcher)
            state = Singleton.get_state()

            # happening on the spawned process (executing task_function in run_job)
            Singleton.set_state(state)

            ret = run_job(
                rehydra_context=self.rehydra_context,
                task_function=self.task_function,
                config=sweep_config,
                job_dir_key="rehydra.sweep.dir",
                job_subdir_key="rehydra.sweep.subdir",
            )
            runs.append(ret)
            # reconfigure the logging subsystem for Rehydra as the run_job call configured it for the Job.
            # This is needed for launchers that calls run_job in the same process and not spawn a new one.
            configure_log(self.config.rehydra.rehydra_logging, self.config.rehydra.verbose)
        return runs
