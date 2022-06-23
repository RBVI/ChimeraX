import sys
import traceback

from . import init

if __name__ == '__main__':
    sess = init(sys.argv)
    try:
        rcode = sess.ui.event_loop()
    except SystemExit as e:
        raise SystemExit(e.code)
    except Exception:
        traceback.print_exc()
        raise SystemExit(os.EX_SOFTWARE)
    raise SystemExit(rcode)
