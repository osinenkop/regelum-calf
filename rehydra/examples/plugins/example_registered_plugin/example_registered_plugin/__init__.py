# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from rehydra.core.plugins import Plugins
from rehydra.plugins.plugin import Plugin


class ExampleRegisteredPlugin(Plugin):
    def __init__(self, v: int) -> None:
        self.v = v

    def add(self, x: int) -> int:
        return self.v + x


def register_example_plugin() -> None:
    """The Rehydra user should call this function before invoking @rehydra.main"""
    Plugins.instance().register(ExampleRegisteredPlugin)
