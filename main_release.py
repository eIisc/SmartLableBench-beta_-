import os
import sys

from main import main as app_main


def _set_release_defaults():
    # Release build must not bundle personal API defaults.
    os.environ["SLB_DEFAULT_API_BASE"] = ""
    os.environ["SLB_DEFAULT_API_KEY"] = ""


def main(argv=None):
    _set_release_defaults()
    return app_main(argv)


if __name__ == "__main__":
    sys.exit(main())
