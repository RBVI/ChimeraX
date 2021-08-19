# vim: set expandtab shiftwidth=4 softtabstop=4:

import getopt
import sys

# Read buildinfo.py without importing chimerax.core because
# this is used with generic python before ChimeraX app is built.
buildinfo = {}
import site, os.path
for spath in site.getsitepackages():
    bi_path = os.path.join(spath, 'chimerax', 'core', 'buildinfo.py')
    if os.path.exists(bi_path):
        exec(open(bi_path).read(), buildinfo)

try:
    opts, args = getopt.getopt(sys.argv[1:], "vypr")
except getopt.GetOptError as e:
    print(e, file=sys.stderr)
    raise SystemExit(1)

for opt, val in opts:
    if opt == "-v":
        # release version
        print(buildinfo['version'])
    elif opt == "-y":
        # copyright year
        year = buildinfo['date'].split('-', 1)[0]
        print(year)
    elif opt == "-p":
        # Windows product version: a,b,c,d with 16-bit integers
        from packaging.version import Version
        version = Version(buildinfo['version'])
        n = len(version.release)
        if n > 4:
            version = version.release[0:4]
        elif n < 4:
            version = list(version.release)
            version.extend([0] * (4 - n))
        print(','.join([str(i) for i in version]))
    elif opt == "-r":
        from packaging.version import Version
        version = Version(buildinfo['version'])
        if version.is_prerelease:
            print('prerelease')
        else:
            print('production')

raise SystemExit(0)
