# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2026 Regents of the University of California. All rights reserved.
# The ChimeraX application is provided pursuant to the ChimeraX license
# agreement, which covers academic and commercial uses. For more details, see
# <https://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
#
# This particular file is part of the ChimeraX library. You can also
# redistribute and/or modify it under the terms of the GNU Lesser General
# Public License version 2.1 as published by the Free Software Foundation.
# For more details, see
# <https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html>
#
# THIS SOFTWARE IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND, EITHER
# EXPRESSED OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
# OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. ADDITIONAL LIABILITY
# LIMITATIONS ARE DESCRIBED IN THE GNU LESSER GENERAL PUBLIC LICENSE
# VERSION 2.1
#
# This notice must be embedded in or attached to all copies, including partial
# copies, of the software or any revisions or derivations thereof.
# === UCSF ChimeraX Copyright ===

"""
Load every shared library under a directory once, and throw them away.

macOS checks a library's code signature the first time it is loaded after
being written, which for a freshly installed app adds up to seconds in the
first programs that run it (the documentation build, and ChimeraX itself).
Run this in the background while there is other work to do, so that cost is
paid off the critical path.  Whether each load succeeds does not matter: the
check happens before the library's dependencies are resolved.

usage: python warm_libraries.py DIRECTORY [SKIP_DIRECTORY_NAME ...]
"""

import os
import sys
from concurrent.futures import ProcessPoolExecutor

MACHO_MAGICS = {
    b"\xcf\xfa\xed\xfe",  # 64-bit
    b"\xca\xfe\xba\xbe",  # universal
}


def libraries(top, skip):
    for root, dirs, files in os.walk(top):
        dirs[:] = [d for d in dirs if d not in skip and d != "__pycache__"]
        for name in files:
            path = os.path.join(root, name)
            if name.endswith((".so", ".dylib")):
                yield path
            elif "." not in name and ".framework" in root and not os.path.islink(path):
                # The main binary of a framework has no extension
                try:
                    with open(path, "rb") as f:
                        if f.read(4) in MACHO_MAGICS:
                            yield path
                except OSError:
                    pass


def load(paths):
    import ctypes

    for path in paths:
        try:
            # RTLD_LOCAL, so nothing loaded here can satisfy a later load
            ctypes.CDLL(path, mode=os.RTLD_LAZY | os.RTLD_LOCAL)
        except OSError:
            pass


def main(top, skip):
    paths = sorted(libraries(top, set(skip)))
    nproc = os.cpu_count() or 1
    chunks = [paths[i::nproc] for i in range(nproc)]
    with ProcessPoolExecutor(nproc) as pool:
        list(pool.map(load, chunks))


if __name__ == "__main__":
    if sys.platform == "darwin" and len(sys.argv) > 1:
        main(sys.argv[1], sys.argv[2:])
