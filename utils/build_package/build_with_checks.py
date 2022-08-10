import sys
import validate
import build
from pytest import ExitCode


def main():
    # pre-build validate
    sys.argv.append("pre")
    exit_code = validate.main()
    if exit_code != ExitCode.OK:
        sys.exit(exit_code)

    # build package
    build.main()

    # post-build validation
    sys.argv.append("post")
    exit_code = validate.main()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
