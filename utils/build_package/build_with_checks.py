import utils.build_package.validate as validate
import build
import post_build


def main():
    validate.main()
    build.main()
    post_build.main()
