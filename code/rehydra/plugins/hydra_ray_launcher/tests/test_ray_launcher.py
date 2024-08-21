# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import sys

from rehydra.core.plugins import Plugins
from rehydra.plugins.launcher import Launcher
from rehydra.test_utils.launcher_common_tests import (
    IntegrationTestSuite,
    LauncherTestSuite,
)
from rehydra.test_utils.test_utils import chdir_plugin_root
from pytest import mark

from rehydra_plugins.rehydra_ray_launcher.ray_launcher import RayLauncher  # type: ignore

chdir_plugin_root()

win_msg = "Ray doesn't support Windows."


@mark.skipif(sys.platform.startswith("win"), reason=win_msg)
def test_discovery() -> None:
    # Tests that this plugin can be discovered via the plugins subsystem when looking for Launchers
    assert RayLauncher.__name__ in [
        x.__name__ for x in Plugins.instance().discover(Launcher)
    ]


@mark.skipif(sys.platform.startswith("win"), reason=win_msg)
@mark.parametrize("launcher_name, overrides", [("ray", [])])
class TestRayLauncher(LauncherTestSuite):
    """
    Run the Launcher test suite on this launcher.
    """

    pass


@mark.skipif(sys.platform.startswith("win"), reason=win_msg)
@mark.parametrize(
    "task_launcher_cfg, extra_flags",
    [
        (
            {},
            [
                "-m",
                "rehydra/launcher=ray",
                "rehydra/rehydra_logging=rehydra_debug",
                "rehydra/job_logging=disabled",
            ],
        )
    ],
)
class TestRayLauncherIntegration(IntegrationTestSuite):
    """
    Run this launcher through the integration test suite.
    """

    pass
