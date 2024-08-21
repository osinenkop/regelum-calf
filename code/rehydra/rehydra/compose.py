# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from textwrap import dedent
from typing import List, Optional

from omegaconf import DictConfig, OmegaConf, open_dict

from rehydra import version
from rehydra.core.global_rehydra import GlobalRehydra
from rehydra.types import RunMode

from ._internal.deprecation_warning import deprecation_warning


def compose(
    config_name: Optional[str] = None,
    overrides: Optional[List[str]] = None,
    return_rehydra_config: bool = False,
    strict: Optional[bool] = None,
) -> DictConfig:
    """
    :param config_name: the name of the config
           (usually the file name without the .yaml extension)
    :param overrides: list of overrides for config file
    :param return_rehydra_config: True to return the rehydra config node in the result
    :param strict: DEPRECATED. If false, returned config has struct mode disabled.
    :return: the composed config
    """

    if overrides is None:
        overrides = []

    assert (
        GlobalRehydra().is_initialized()
    ), "GlobalRehydra is not initialized, use @rehydra.main() or call one of the rehydra initialization methods first"

    gh = GlobalRehydra.instance()
    assert gh.rehydra is not None
    cfg = gh.rehydra.compose_config(
        config_name=config_name,
        overrides=overrides,
        run_mode=RunMode.RUN,
        from_shell=False,
        with_log_configuration=False,
    )
    assert isinstance(cfg, DictConfig)

    if not return_rehydra_config:
        if "rehydra" in cfg:
            with open_dict(cfg):
                del cfg["rehydra"]

    if strict is not None:
        if version.base_at_least("1.2"):
            raise TypeError("got an unexpected 'strict' argument")
        else:
            deprecation_warning(
                dedent(
                    """
                    The strict flag in the compose API is deprecated.
                    See https://rehydra.cc/docs/1.2/upgrades/0.11_to_1.0/strict_mode_flag_deprecated for more info.
                    """
                )
            )
            OmegaConf.set_struct(cfg, strict)

    return cfg
