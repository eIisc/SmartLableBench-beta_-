import os
import sys

from main import main as app_main


def _set_test_defaults():
    # Test build keeps the pre-filled API settings.
    os.environ["SLB_DEFAULT_API_BASE"] = "https://api.deepseek.com"
    os.environ["SLB_DEFAULT_API_KEY"] = "sk-32d8d385d45346af973538a9e026f63d"


def main(argv=None):
    _set_test_defaults()
    return app_main(argv)


if __name__ == "__main__":
    sys.exit(main())
