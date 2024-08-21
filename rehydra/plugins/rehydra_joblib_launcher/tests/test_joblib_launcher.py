# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from typing import Any

from rehydra.core.plugins import Plugins
from rehydra.plugins.launcher import Launcher
from rehydra.test_utils.launcher_common_tests import (
    IntegrationTestSuite,
    LauncherTestSuite,
)
from rehydra.test_utils.test_utils import TSweepRunner, chdir_plugin_root
from pytest import mark

from rehydra_plugins.rehydra_joblib_launcher.joblib_launcher import JoblibLauncher

chdir_plugin_root()


def test_discovery() -> None:
    # Tests that this plugin can be discovered via the plugins subsystem when looking for Launchers
    assert JoblibLauncher.__name__ in [
        x.__name__ for x in Plugins.instance().discover(Launcher)
    ]


@mark.parametrize("launcher_name, overrides", [("joblib", [])])
class TestJoblibLauncher(LauncherTestSuite):
    """
    Run the Launcher test suite on this launcher.
    """

    pass


@mark.parametrize(
    "task_launcher_cfg, extra_flags",
    [
        # joblib with process-based backend (default)
        (
            {},
            [
                "-m",
                "rehydra/job_logging=rehydra_debug",
                "rehydra/job_logging=disabled",
                "rehydra/launcher=joblib",
            ],
        )
    ],
)
class TestJoblibLauncherIntegration(IntegrationTestSuite):
    """
    Run this launcher through the integration test suite.
    """

    pass


def test_example_app(rehydra_sweep_runner: TSweepRunner, tmpdir: Any) -> None:
    with rehydra_sweep_runner(
        calling_file="example/my_app.py",
        calling_module=None,
        task_function=None,
        config_path=".",
        config_name="config",
        overrides=["task=1,2,3,4", f"rehydra.sweep.dir={tmpdir}"],
    ) as sweep:
        overrides = {("task=1",), ("task=2",), ("task=3",), ("task=4",)}

        assert sweep.returns is not None and len(sweep.returns[0]) == 4
        for ret in sweep.returns[0]:
            assert tuple(ret.overrides) in overrides


@mark.parametrize(
    "overrides",
    [
        "rehydra.launcher.batch_size=1",
        "rehydra.launcher.max_nbytes=10000",
        "rehydra.launcher.max_nbytes=1M",
        "rehydra.launcher.pre_dispatch=all",
        "rehydra.launcher.pre_dispatch=10",
        "rehydra.launcher.pre_dispatch=3*n_jobs",
    ],
)
def test_example_app_launcher_overrides(
    rehydra_sweep_runner: TSweepRunner, overrides: str
) -> None:
    with rehydra_sweep_runner(
        calling_file="example/my_app.py",
        calling_module=None,
        task_function=None,
        config_path=".",
        config_name="config",
        overrides=[overrides],
    ) as sweep:
        assert sweep.returns is not None and len(sweep.returns[0]) == 1
