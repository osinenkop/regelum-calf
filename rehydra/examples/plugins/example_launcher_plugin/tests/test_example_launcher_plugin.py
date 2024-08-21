# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from pytest import mark

from rehydra.core.plugins import Plugins
from rehydra.plugins.launcher import Launcher
from rehydra.test_utils.launcher_common_tests import (
    IntegrationTestSuite,
    LauncherTestSuite,
)


from rehydra_plugins.example_launcher_plugin.example_launcher import ExampleLauncher


def test_discovery() -> None:
    # Tests that this plugin can be discovered via the plugins subsystem when looking for Launchers
    assert ExampleLauncher.__name__ in [
        x.__name__ for x in Plugins.instance().discover(Launcher)
    ]


@mark.parametrize("launcher_name, overrides", [("example", [])])
class TestExampleLauncher(LauncherTestSuite):
    """
    Run the Launcher test suite on this launcher.
    Note that rehydra/launcher/example.yaml should be provided by this launcher.
    """

    pass


@mark.parametrize(
    "task_launcher_cfg, extra_flags",
    [({}, ["-m", "rehydra/launcher=example"])],
)
class TestExampleLauncherIntegration(IntegrationTestSuite):
    """
    Run this launcher through the integration test suite.
    """

    pass
