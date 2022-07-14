import sys
from . import init

if __name__ == '__main__':
    exit_code = init(sys.argv)
    raise SystemExit(exit_code)
