from typing import List, Dict, Any
import pathlib
import versioningit


_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent.parent.parent

versioningit_config = {
    "format": {
        "distance": "{base_version}+{distance}.{vcs}{rev}",
        "dirty": "{base_version}+{distance}.{vcs}{rev}.dirty",
        "distance-dirty": "{base_version}+{distance}.{vcs}{rev}.dirty",
    },
    "write": {
        "file": "src/espresso/_version.py"
    },
    "tag2version": {
        "rmprefix": "v",
        "rmsuffix": "-build",
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

    try:            # generate _version.py from VCS
        versioningit.get_version(_ROOT, versioningit_config, True)
    except:
        pass        # VCS not found, there should be a version file to read

    # read src/espresso/_version.py
    with open(str(_ROOT / "src" / "espresso" / "_version.py")) as f:
        for line in f:
            if line.startswith("__version__ = "):
                _, _, version = line.partition("=")
                return version.strip(" \n'\"")
    raise RuntimeError("unable to read the version from src/espresso/_version.py")


__all__ = ["dynamic_metadata"]


def __dir__() -> List[str]:
    return __all__
