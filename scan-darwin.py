# vi: set expandtab shiftwidth=4 softtabstop=4:
#
# On Darwin, scan build tree for shared libraries and bundles
# and check that the install names are correct.  If they are
# not correct, output that information so the Makefile that
# creates the file can use install_name_tool to fix it.
#
# Shouldn't have any problems with programs if the shared
# libraries are named correctly.

from __future__ import print_function

import os
import subprocess
import sys

if not sys.platform.startswith('darwin'):
    print('only works on Mac OS X', file=sys.stderr)
    raise SystemExit(1)

def check_otool(filename, rpath=False, program=False):
    output = subprocess.check_output(['/usr/bin/otool', '-L', filename])
                                     # stderr=subprocess.DEVNULL)
    lines = [x.lstrip().split(None, 1)[0] for x in output.split('\n') if x]
    # first line is filename, second line is library's install name,
    # subsequent lines are libraries linked with
    if program or filename.endswith('.so'):
        libraries = lines[1:]
    else:
        name = lines[1]
        libraries = lines[2:]
        if rpath and not name.startswith('@rpath'):
            print('error:', filename, 'is not named @rpath/...')
        elif not name[0] == '@':
            print('error:', filename, 'does not have a relative name')
    for lib in libraries:
        if (lib.startswith('@') or lib.startswith('/usr/lib/')
                or lib.startswith('/System/')):
            continue
        print('warning:', filename, 'check for', lib)


for dirpath, dirnames, filenames in os.walk('build'):
    if dirpath == 'build':
        dirnames.remove('tmp')
    filenames = [fn for fn in filenames if fn.endswith(('.dylib', '.so'))]
    if not filenames:
        continue
    if dirpath == 'build/lib':
        for fn in filenames:
            check_otool(os.path.join(dirpath, fn), rpath=True)
        continue
    if dirpath == 'build/bin':
        for fn in filenames:
            check_otool(os.path.join(dirpath, fn), program=True)
        continue
    # normal check
    for fn in filenames:
        check_otool(os.path.join(dirpath, fn))
