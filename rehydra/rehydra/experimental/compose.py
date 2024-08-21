# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from typing import List, Optional

from omegaconf import DictConfig

from rehydra import version
from rehydra._internal.deprecation_warning import deprecation_warning


def compose(
    config_name: Optional[str] = None,
    overrides: List[str] = [],
    return_rehydra_config: bool = False,
    strict: Optional[bool] = None,
) -> DictConfig:
    from rehydra import compose as real_compose

    message = (
        "rehydra.experimental.compose() is no longer experimental. Use rehydra.compose()"
    )

    if version.base_at_least("1.2"):
        raise ImportError(message)

    deprecation_warning(message=message)
    return real_compose(
        config_name=config_name,
        overrides=overrides,
        return_rehydra_config=return_rehydra_config,
        strict=strict,
    )
