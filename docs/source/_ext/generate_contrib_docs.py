from pathlib import Path

import cofi_espresso as esp
from cofi_espresso._machine import docutils 

# APV removed following as it is never used(?)
# CONTRIBS = [nm for nm in dir(esp) if not nm.startswith("_") and nm[0].islower()]
BASE_PATH = esp.__path__[0]
DEST_PATH = Path(__file__).resolve().parent.parent / "user_guide" / "contrib" / "generated"

def setup(app):
    app.connect("builder-inited", lambda _: docutils.gen_contrib_docs(BASE_PATH, DEST_PATH))
    return {
        'version': '0.1',
        'parallel_read_safe': True,
        'parallel_write_safe': True,
    }
