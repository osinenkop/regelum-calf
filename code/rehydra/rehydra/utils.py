# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import logging.config
import os
from pathlib import Path
from typing import Any, Callable

import rehydra._internal.instantiate._instantiate2
import rehydra.types
from rehydra._internal.utils import _locate
from rehydra.core.rehydra_config import RehydraConfig

log = logging.getLogger(__name__)

# Instantiation related symbols
instantiate = rehydra._internal.instantiate._instantiate2.instantiate
call = instantiate
ConvertMode = rehydra.types.ConvertMode


def get_class(path: str) -> type:
    """
    Look up a class based on a dotpath.
    Fails if the path does not point to a class.

    >>> import my_module
    >>> from rehydra.utils import get_class
    >>> assert get_class("my_module.MyClass") is my_module.MyClass
    """
    try:
        cls = _locate(path)
        if not isinstance(cls, type):
            raise ValueError(
                f"Located non-class of type '{type(cls).__name__}'"
                + f" while loading '{path}'"
            )
        return cls
    except Exception as e:
        log.error(f"Error getting class at {path}: {e}")
        raise e


def get_method(path: str) -> Callable[..., Any]:
    """
    Look up a callable based on a dotpath.
    Fails if the path does not point to a callable object.

    >>> import my_module
    >>> from rehydra.utils import get_method
    >>> assert get_method("my_module.my_function") is my_module.my_function
    """
    try:
        obj = _locate(path)
        if not callable(obj):
            raise ValueError(
                f"Located non-callable of type '{type(obj).__name__}'"
                + f" while loading '{path}'"
            )
        cl: Callable[..., Any] = obj
        return cl
    except Exception as e:
        log.error(f"Error getting callable at {path} : {e}")
        raise e


# Alias for get_method
get_static_method = get_method


def get_object(path: str) -> Any:
    """
    Look up an entity based on the dotpath.
    Does not perform any type checks on the entity.

    >>> import my_module
    >>> from rehydra.utils import get_object
    >>> assert get_object("my_module.my_object") is my_module.my_object
    """
    try:
        obj = _locate(path)
        return obj
    except Exception as e:
        log.error(f"Error getting object at {path} : {e}")
        raise e


def get_original_cwd() -> str:
    """
    :return: the original working directory the Rehydra application was launched from
    """
    if not RehydraConfig.initialized():
        raise ValueError(
            "get_original_cwd() must only be used after RehydraConfig is initialized"
        )
    ret = RehydraConfig.get().runtime.cwd
    assert ret is not None and isinstance(ret, str)
    return ret


def to_absolute_path(path: str) -> str:
    """
    converts the specified path to be absolute path.
    if the input path is relative, it's interpreted as relative to the original working directory
    if it's absolute, it's returned as is
    :param path: path to convert
    :return:
    """
    p = Path(path)
    if not RehydraConfig.initialized():
        base = Path(os.getcwd())
    else:
        base = Path(get_original_cwd())
    if p.is_absolute():
        ret = p
    else:
        ret = base / p
    return str(ret)
