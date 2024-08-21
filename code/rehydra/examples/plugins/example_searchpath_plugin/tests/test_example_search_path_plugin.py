# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from rehydra.core.global_rehydra import GlobalRehydra
from rehydra.core.plugins import Plugins
from rehydra import initialize
from rehydra.plugins.search_path_plugin import SearchPathPlugin

from rehydra_plugins.example_searchpath_plugin.example_searchpath_plugin import (
    ExampleSearchPathPlugin,
)


def test_discovery() -> None:
    # Tests that this plugin can be discovered via the plugins subsystem when looking at all Plugins
    assert ExampleSearchPathPlugin.__name__ in [
        x.__name__ for x in Plugins.instance().discover(SearchPathPlugin)
    ]


def test_config_installed() -> None:
    with initialize(version_base=None):
        config_loader = GlobalRehydra.instance().config_loader()
        assert "my_default_output_dir" in config_loader.get_group_options(
            "rehydra/output"
        )
