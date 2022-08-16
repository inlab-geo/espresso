import sys
import subprocess
from pathlib import Path
from pytest import ExitCode

def main():
    validate_script = str(Path(__file__).parent / "validate.py")
    build_script = str(Path(__file__).parent / "build.py")

    # pre-build validate
    exit_code = subprocess.call([sys.executable, validate_script, "pre"])
    if exit_code != ExitCode.OK:
        sys.exit(exit_code)

    # build package
    exit_code = subprocess.call([sys.executable, build_script])
    if exit_code:
        sys.exit(exit_code)

    # post-build validation
    exit_code = subprocess.call([sys.executable, validate_script, "post"])
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
