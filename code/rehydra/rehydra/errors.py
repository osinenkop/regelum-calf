# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from typing import Optional, Sequence


class RehydraException(Exception):
    ...


class CompactRehydraException(RehydraException):
    ...


class OverrideParseException(CompactRehydraException):
    def __init__(self, override: str, message: str) -> None:
        super(OverrideParseException, self).__init__(message)
        self.override = override
        self.message = message


class InstantiationException(CompactRehydraException):
    ...


class ConfigCompositionException(CompactRehydraException):
    ...


class SearchPathException(CompactRehydraException):
    ...


class MissingConfigException(IOError, ConfigCompositionException):
    def __init__(
        self,
        message: str,
        missing_cfg_file: Optional[str] = None,
        options: Optional[Sequence[str]] = None,
    ) -> None:
        super(MissingConfigException, self).__init__(message)
        self.missing_cfg_file = missing_cfg_file
        self.options = options


class RehydraDeprecationError(RehydraException):
    ...
