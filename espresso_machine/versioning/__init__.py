from typing import List, Dict, Any
import pathlib
import versioningit


_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent

versioningit_config = {
    "format": {
        "distance": "{base_version}+{distance}.{vcs}{rev}.core",
        "dirty": "{base_version}+{distance}.{vcs}{rev}.dirty.core",
        "distance-dirty": "{base_version}+{distance}.{vcs}{rev}.dirty.core",
    },
    "write": {
        "file": "src/espresso/_version.py"
    },
    "tag2version": {
        "rmprefix": "v",
        "rmsuffix": "",
    }
}


def dynamic_metadata(
    field: str,
    settings: Dict[str, Any] = None,
) -> str:
    if field != "version":
        msg = "Only the 'version' field is supported"
        raise ValueError(msg)

    if settings:
        msg = "No inline configuration is supported"
        raise ValueError(msg)

    # generate src/espresso/_version.py
    return versioningit.get_version(_ROOT, versioningit_config, True)


__all__ = ["dynamic_metadata"]


def __dir__() -> List[str]:
    return __all__
