# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import copy
import logging
import os
import re
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from os.path import splitext
from pathlib import Path
from textwrap import dedent
from typing import Any, Dict, Optional, Sequence, Union, cast

from omegaconf import DictConfig, OmegaConf, open_dict, read_write

from rehydra import version
from rehydra._internal.deprecation_warning import deprecation_warning
from rehydra.core.rehydra_config import RehydraConfig
from rehydra.core.singleton import Singleton
from rehydra.types import RehydraContext, TaskFunction

import rich.logging

log = logging.getLogger(__name__)


def simple_stdout_log_config(level: int = logging.INFO) -> None:
    root = logging.getLogger()
    root.setLevel(level)
    #handler = logging.StreamHandler(sys.stdout)
    #formatter = logging.Formatter("%(message)s")
    #handler.setFormatter(formatter)
    #root.addHandler(handler)


def configure_log(
    log_config: DictConfig,
    verbose_config: Union[bool, str, Sequence[str]] = False,
) -> None:
    assert isinstance(verbose_config, (bool, str)) or OmegaConf.is_list(verbose_config)
    if log_config is not None:
        conf: Dict[str, Any] = OmegaConf.to_container(  # type: ignore
            log_config, resolve=True
        )
        if conf["root"] is not None:
            conf["handlers"]["console"] = {"class": 'rich.logging.RichHandler'}
            logging.config.dictConfig(conf)
    else:
        # default logging to stdout
        root = logging.getLogger()
        root.setLevel(logging.INFO)
        #handler = logging.StreamHandler(sys.stdout)
        #formatter = logging.Formatter(
        #    "[%(asctime)s][%(name)s][%(levelname)s] - %(message)s"
        #)
        #handler.setFormatter(formatter)
        #root.addHandler(handler)
    if isinstance(verbose_config, bool):
        if verbose_config:
            logging.getLogger().setLevel(logging.DEBUG)
    else:
        if isinstance(verbose_config, str):
            verbose_list = OmegaConf.create([verbose_config])
        elif OmegaConf.is_list(verbose_config):
            verbose_list = verbose_config  # type: ignore
        else:
            assert False

        for logger in verbose_list:
            logging.getLogger(logger).setLevel(logging.DEBUG)


def _save_config(cfg: DictConfig, filename: str, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(str(output_dir / filename), "w", encoding="utf-8") as file:
        file.write(OmegaConf.to_yaml(cfg))


def filter_overrides(overrides: Sequence[str]) -> Sequence[str]:
    """
    :param overrides: overrides list
    :return: returning a new overrides list with all the keys starting with rehydra. filtered.
    """
    return [x for x in overrides if not x.startswith("rehydra.")]


def _check_rehydra_context(rehydra_context: Optional[RehydraContext]) -> None:
    if rehydra_context is None:
        # rehydra_context is required as of Rehydra 1.2.
        # We can remove this check in Rehydra 1.3.
        raise TypeError(
            dedent(
                """
                run_job's signature has changed: the `rehydra_context` arg is now required.
                For more info, check https://github.com/facebookresearch/rehydra/pull/1581."""
            ),
        )


def run_job(
    task_function: TaskFunction,
    config: DictConfig,
    job_dir_key: str,
    job_subdir_key: Optional[str],
    rehydra_context: RehydraContext,
    configure_logging: bool = True,
) -> "JobReturn":
    _check_rehydra_context(rehydra_context)
    callbacks = rehydra_context.callbacks

    old_cwd = os.getcwd()
    orig_rehydra_cfg = RehydraConfig.instance().cfg

    # init Rehydra config for config evaluation
    RehydraConfig.instance().set_config(config)

    output_dir = str(OmegaConf.select(config, job_dir_key))
    if job_subdir_key is not None:
        # evaluate job_subdir_key lazily.
        # this is running on the client side in sweep and contains things such as job:id which
        # are only available there.
        subdir = str(OmegaConf.select(config, job_subdir_key))
        output_dir = os.path.join(output_dir, subdir)

    with read_write(config.rehydra.runtime):
        with open_dict(config.rehydra.runtime):
            config.rehydra.runtime.output_dir = os.path.abspath(output_dir)

    # update Rehydra config
    RehydraConfig.instance().set_config(config)
    _chdir = None
    try:
        ret = JobReturn()
        task_cfg = copy.deepcopy(config)
        with read_write(task_cfg):
            with open_dict(task_cfg):
                del task_cfg["rehydra"]

        ret.cfg = task_cfg
        rehydra_cfg = copy.deepcopy(RehydraConfig.instance().cfg)
        assert isinstance(rehydra_cfg, DictConfig)
        ret.rehydra_cfg = rehydra_cfg
        overrides = OmegaConf.to_container(config.rehydra.overrides.task)
        assert isinstance(overrides, list)
        ret.overrides = overrides
        # handle output directories here
        Path(str(output_dir)).mkdir(parents=True, exist_ok=True)

        _chdir = rehydra_cfg.rehydra.job.chdir

        if _chdir is None:
            if version.base_at_least("1.2"):
                _chdir = False

        if _chdir is None:
            url = "https://rehydra.cc/docs/1.2/upgrades/1.1_to_1.2/changes_to_job_working_dir/"
            deprecation_warning(
                message=dedent(
                    f"""\
                    Future Rehydra versions will no longer change working directory at job runtime by default.
                    See {url} for more information."""
                ),
                stacklevel=2,
            )
            _chdir = True

        if _chdir:
            os.chdir(output_dir)
            ret.working_dir = output_dir
        else:
            ret.working_dir = os.getcwd()

        if configure_logging:
            configure_log(config.rehydra.job_logging, config.rehydra.verbose)

        if config.rehydra.output_subdir is not None:
            rehydra_output = Path(config.rehydra.runtime.output_dir) / Path(
                config.rehydra.output_subdir
            )
            _save_config(task_cfg, "config.yaml", rehydra_output)
            _save_config(rehydra_cfg, "rehydra.yaml", rehydra_output)
            _save_config(config.rehydra.overrides.task, "overrides.yaml", rehydra_output)

        with env_override(rehydra_cfg.rehydra.job.env_set):
            callbacks.on_job_start(config=config, task_function=task_function)
            try:
                ret.return_value = task_function(task_cfg)
                ret.status = JobStatus.COMPLETED
            except Exception as e:
                ret.return_value = e
                ret.status = JobStatus.FAILED

        ret.task_name = JobRuntime.instance().get("name")

        _flush_loggers()

        callbacks.on_job_end(config=config, job_return=ret)

        return ret
    finally:
        RehydraConfig.instance().cfg = orig_rehydra_cfg
        if _chdir:
            os.chdir(old_cwd)


def get_valid_filename(s: str) -> str:
    s = str(s).strip().replace(" ", "_")
    return re.sub(r"(?u)[^-\w.]", "", s)


def setup_globals() -> None:
    # please add documentation when you add a new resolver
    OmegaConf.register_new_resolver(
        "now",
        lambda pattern: datetime.now().strftime(pattern),
        use_cache=True,
        replace=True,
    )
    OmegaConf.register_new_resolver(
        "rehydra",
        lambda path: OmegaConf.select(cast(DictConfig, RehydraConfig.get()), path),
        replace=True,
    )

    vi = sys.version_info
    version_dict = {
        "major": f"{vi[0]}",
        "minor": f"{vi[0]}.{vi[1]}",
        "micro": f"{vi[0]}.{vi[1]}.{vi[2]}",
    }
    OmegaConf.register_new_resolver(
        "python_version", lambda level="minor": version_dict.get(level), replace=True
    )


class JobStatus(Enum):
    UNKNOWN = 0
    COMPLETED = 1
    FAILED = 2


@dataclass
class JobReturn:
    overrides: Optional[Sequence[str]] = None
    cfg: Optional[DictConfig] = None
    rehydra_cfg: Optional[DictConfig] = None
    working_dir: Optional[str] = None
    task_name: Optional[str] = None
    status: JobStatus = JobStatus.UNKNOWN
    _return_value: Any = None

    @property
    def return_value(self) -> Any:
        assert self.status != JobStatus.UNKNOWN, "return_value not yet available"
        if self.status == JobStatus.COMPLETED:
            return self._return_value
        else:
            sys.stderr.write(
                f"Error executing job with overrides: {self.overrides}" + os.linesep
            )
            raise self._return_value

    @return_value.setter
    def return_value(self, value: Any) -> None:
        self._return_value = value


class JobRuntime(metaclass=Singleton):
    def __init__(self) -> None:
        self.conf: DictConfig = OmegaConf.create()
        self.set("name", "UNKNOWN_NAME")

    def get(self, key: str) -> Any:
        ret = OmegaConf.select(self.conf, key)
        if ret is None:
            raise KeyError(f"Key not found in {type(self).__name__}: {key}")
        return ret

    def set(self, key: str, value: Any) -> None:
        log.debug(f"Setting {type(self).__name__}:{key}={value}")
        self.conf[key] = value


def validate_config_path(config_path: Optional[str]) -> None:
    if config_path is not None:
        split_file = splitext(config_path)
        if split_file[1] in (".yaml", ".yml"):
            msg = dedent(
                """\
            Using config_path to specify the config name is not supported, specify the config name via config_name.
            See https://rehydra.cc/docs/1.2/upgrades/0.11_to_1.0/config_path_changes
            """
            )
            raise ValueError(msg)


@contextmanager
def env_override(env: Dict[str, str]) -> Any:
    """Temporarily set environment variables inside the context manager and
    fully restore previous environment afterwards
    """
    original_env = {key: os.getenv(key) for key in env}
    os.environ.update(env)
    try:
        yield
    finally:
        for key, value in original_env.items():
            if value is None:
                del os.environ[key]
            else:
                os.environ[key] = value


def _flush_loggers() -> None:
    # Python logging does not have an official API to flush all loggers.
    # This will have to do.
    for h_weak_ref in logging._handlerList:  # type: ignore
        try:
            h_weak_ref().flush()
        except Exception:
            # ignore exceptions thrown during flushing
            pass
