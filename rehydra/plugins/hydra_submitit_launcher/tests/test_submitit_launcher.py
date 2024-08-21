# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from pathlib import Path
from typing import Type

from rehydra.core.plugins import Plugins
from rehydra.plugins.launcher import Launcher
from rehydra.test_utils.launcher_common_tests import (
    IntegrationTestSuite,
    LauncherTestSuite,
)
from rehydra.test_utils.test_utils import chdir_plugin_root, run_python_script
from pytest import mark

from rehydra_plugins.rehydra_submitit_launcher import submitit_launcher

chdir_plugin_root()


@mark.parametrize(
    "cls", [submitit_launcher.LocalLauncher, submitit_launcher.SlurmLauncher]
)
def test_discovery(cls: Type[Launcher]) -> None:
    # Tests that this plugin can be discovered via the plugins subsystem when looking for Launchers
    assert cls.__name__ in [x.__name__ for x in Plugins.instance().discover(Launcher)]


@mark.parametrize(
    "launcher_name, overrides", [("submitit_local", ["rehydra.launcher.timeout_min=2"])]
)
class TestSubmititLauncher(LauncherTestSuite):
    pass


@mark.parametrize(
    "task_launcher_cfg, extra_flags",
    [
        (
            {},
            [
                "-m",
                "rehydra/rehydra_logging=disabled",
                "rehydra/job_logging=disabled",
                "rehydra/launcher=submitit_local",
                "rehydra.launcher.gpus_per_node=0",
                "rehydra.launcher.timeout_min=1",
            ],
        ),
    ],
)
class TestSubmititLauncherIntegration(IntegrationTestSuite):
    pass


def test_example(tmpdir: Path) -> None:
    run_python_script(
        [
            "example/my_app.py",
            "-m",
            f"rehydra.sweep.dir={tmpdir}",
            "rehydra/launcher=submitit_local",
        ]
    )
