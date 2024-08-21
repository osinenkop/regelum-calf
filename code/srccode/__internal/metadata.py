from types import MappingProxyType

from srccode import SrccodeBase


class Metadata(SrccodeBase):
    def __init__(self, content):
        self.content = MappingProxyType(content)

    def __enter__(self):
        self._metadata = self.content

    def __exit__(self, exc_type, exc_val, exc_tb):
        delattr(SrccodeBase, f"_{SrccodeBase.__name__}__metadata")
