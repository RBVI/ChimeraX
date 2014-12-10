#!/bin/env python
# vim: set expandtab shiftwidth=4 softtabstop=4:

# --- UCSF Chimera Copyright ---
# Copyright (c) 2000-2014 Regents of the University of California.
# All rights reserved.  This software provided pursuant to a
# license agreement containing restrictions on its disclosure,
# duplication and use.  This notice must be embedded in or
# attached to all copies, including partial copies, of the
# software or any revisions or derivations thereof.
# --- UCSF Chimera Copyright ---

import os
import sys


class Module:
    def __init__(self, path):
        self.__path__ = path
        contents = open(path, "rb")
        self.__code__ = compile(contents.read(), path, 'exec')


def main(argv):
    if not sys.argv[2:]:
        print('usage:', sys.argv[0], 'Python-freeze-src-dir file.py ...')
        raise SystemExit(2)

    freeze_src = argv[1]
    sys.path.insert(0, freeze_src)
    import makefreeze

    modules = {}
    for arg in argv[2:]:
        base = os.path.basename(arg)
        mod, ext = os.path.splitext(base)
        modules[mod] = Module(arg)
    makefreeze.makefreeze('./', modules, debug=1)

if __name__ == '__main__':
    main(sys.argv)
