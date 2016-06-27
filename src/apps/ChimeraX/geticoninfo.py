# vim: set expandtab shiftwidth=4 softtabstop=4:

import getopt
import sys

from chimerax.core import buildinfo

try:
    opts, args = getopt.getopt(sys.argv[1:], "vpy")
except getopt.GetOptError as e:
    print(e, file=sys.stderr)
    raise SystemExit(1)

for opt, val in opts:
    if opt == "-v":
        # release version
        print("0.1")
    elif opt == "-y":
        # copyright year
        year = buildinfo.date.split('-', 1)[0]
        print(year)
    elif opt == "-p":
        # Windows product version: a,b,c,d
        version = [0, 1]
        n = len(version)
        if n > 4:
            version = version[0:4]
        elif n < 4:
            version.extend([0] * (4 - n))
        print(','.join([str(i) for i in version]))

raise SystemExit(0)
