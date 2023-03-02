from pathlib import Path
import importlib
import sys

import cofi_espresso as esp


# APV removed following as it is never used(?)
# CONTRIBS = [nm for nm in dir(esp) if not nm.startswith("_") and nm[0].islower()]
BASE_PATH = esp.__path__[0]
DEST_PATH = Path(__file__).resolve().parent.parent / "user_guide" / "contrib" / "generated"
ROOT_PATH = Path(__file__).resolve().parent.parent.parent.parent
_tools_doc_utils_gen_docs = ROOT_PATH / "tools" / "doc_utils"

try:
    sys.path.append(str(_tools_doc_utils_gen_docs))
    import gen_docs
except:
    from esp._machine import gen_docs           # TODO test this after build.py is finished


def setup(app):
    app.connect("builder-inited", lambda _: gen_docs.gen_contrib_docs(BASE_PATH, DEST_PATH))
    return {
        'version': '0.1',
        'parallel_read_safe': True,
        'parallel_write_safe': True,
    }
