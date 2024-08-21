# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import copy
from typing import Any, Optional

from rehydra import version
from rehydra._internal.deprecation_warning import deprecation_warning
from rehydra.core.global_rehydra import GlobalRehydra
from rehydra.core.singleton import Singleton
from rehydra.initialize import _UNSPECIFIED_


def get_gh_backup() -> Any:
    if GlobalRehydra in Singleton._instances:
        return copy.deepcopy(Singleton._instances[GlobalRehydra])
    else:
        return None


def restore_gh_from_backup(_gh_backup: Any) -> Any:
    if _gh_backup is None:
        del Singleton._instances[GlobalRehydra]
    else:
        Singleton._instances[GlobalRehydra] = _gh_backup


class initialize:
    def __init__(
        self,
        config_path: Optional[str] = _UNSPECIFIED_,
        job_name: Optional[str] = None,
        caller_stack_depth: int = 1,
    ) -> None:
        from rehydra import initialize as real_initialize

        message = (
            "rehydra.experimental.initialize() is no longer experimental. "
            "Use rehydra.initialize()"
        )

        if version.base_at_least("1.2"):
            raise ImportError(message)

        deprecation_warning(message=message)

        self.delegate = real_initialize(
            config_path=config_path,
            job_name=job_name,
            caller_stack_depth=caller_stack_depth + 1,
        )

    def __enter__(self, *args: Any, **kwargs: Any) -> None:
        self.delegate.__enter__(*args, **kwargs)

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.delegate.__exit__(exc_type, exc_val, exc_tb)

    def __repr__(self) -> str:
        return "rehydra.experimental.initialize()"


class initialize_config_module:
    """
    Initializes Rehydra and add the config_module to the config search path.
    The config module must be importable (an __init__.py must exist at its top level)
    :param config_module: absolute module name, for example "foo.bar.conf".
    :param job_name: the value for rehydra.job.name (default is 'app')
    """

    def __init__(self, config_module: str, job_name: str = "app") -> None:
        from rehydra import initialize_config_module as real_initialize_config_module

        message = (
            "rehydra.experimental.initialize_config_module() is no longer experimental. "
            "Use rehydra.initialize_config_module()."
        )

        if version.base_at_least("1.2"):
            raise ImportError(message)

        deprecation_warning(message=message)

        self.delegate = real_initialize_config_module(
            config_module=config_module, job_name=job_name
        )

    def __enter__(self, *args: Any, **kwargs: Any) -> None:
        self.delegate.__enter__(*args, **kwargs)

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.delegate.__exit__(exc_type, exc_val, exc_tb)

    def __repr__(self) -> str:
        return "rehydra.experimental.initialize_config_module()"


class initialize_config_dir:
    """
    Initializes Rehydra and add an absolute config dir to the to the config search path.
    The config_dir is always a path on the file system and is must be an absolute path.
    Relative paths will result in an error.
    :param config_dir: absolute file system path
    :param job_name: the value for rehydra.job.name (default is 'app')
    """

    def __init__(self, config_dir: str, job_name: str = "app") -> None:
        from rehydra import initialize_config_dir as real_initialize_config_dir

        message = (
            "rehydra.experimental.initialize_config_dir() is no longer experimental. "
            "Use rehydra.initialize_config_dir()."
        )

        if version.base_at_least("1.2"):
            raise ImportError(message)

        deprecation_warning(message=message)

        self.delegate = real_initialize_config_dir(
            config_dir=config_dir, job_name=job_name
        )

    def __enter__(self, *args: Any, **kwargs: Any) -> None:
        self.delegate.__enter__(*args, **kwargs)

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.delegate.__exit__(exc_type, exc_val, exc_tb)

    def __repr__(self) -> str:
        return "rehydra.experimental.initialize_config_dir()"
