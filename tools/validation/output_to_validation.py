from shutil import copytree
from pathlib import Path

current_dir = Path(__file__).resolve().parent
root_dir = current_dir.parent.parent
NOTEBOOKS = "notebooks"
NOTEBOOKS_DIR = str(root_dir / NOTEBOOKS)
OUTPUT = "_output"
OUTPUT_DIR = str(root_dir / "utils" / "validation" / OUTPUT)
VALIDATION = "_validation_output"
VALIDATION_DIR = str(root_dir / "utils" / "validation" / VALIDATION)

if __name__ == "__main__":
    copytree(OUTPUT_DIR, VALIDATION_DIR, dirs_exist_ok=True)
