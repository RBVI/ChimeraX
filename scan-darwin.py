# vi: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2016 Regents of the University of California.
# All rights reserved.  This software provided pursuant to a
# license agreement containing restrictions on its disclosure,
# duplication and use.  For details see:
# http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html
# This notice must be embedded in or attached to all copies,
# including partial copies, of the software or any revisions
# or derivations thereof.
# === UCSF ChimeraX Copyright ===

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

# Libraries in the following directories are safe
SYSTEM_LIBDIRS = (
    '@', '/usr/lib/', '/System/', '/Library',
)
# except for the following
OVERRIDE_LIBS = (
    '/System/Library/Frameworks/Python.framework/',
    '/Library/Frameworks/Python.framework/',
)

# speed up processing by ignoring files with the following
IGNORE_SUFFIXES = (
    '.a', '.o', '.py', '.pyc', '.cpp', '.h', '.cif', '.pdb', '.stl',
)

if not sys.platform.startswith('darwin'):
    print('only works on Mac OS X', file=sys.stderr)
    raise SystemExit(1)

directory = None
if len(sys.argv) == 2:
    directory = sys.argv[1]
if directory is None:
    print('usage: %s directory' % sys.argv[0], file=sys.stderr)
    raise SystemExit(2)


def check_otool(filename, rpath=False, program=False):
    output = subprocess.check_output(['/usr/bin/otool', '-l', filename])
    lines = [x.lstrip().split(None, 1) for x in output.split('\n') if x]
    if not lines or len(lines) < 2:
        return
    # first line is filename, second line is library's install name,
    # subsequent lines are libraries linked with
    cmd = None
    for line in lines:
        if line[0] == 'cmd':
            cmd = line[1]
            continue
        if cmd == 'LC_ID_DYLIB' and line[0] == 'name':
            name = line[1].rsplit(None, 2)[0]
            if rpath and not name.startswith('@rpath'):
                print('error:', filename, 'is not named @rpath/...:', name)
            elif not name[0] == '@':
                print('error:', filename, 'does not have a relative name:',
                      name)
            cmd = None
        elif cmd == 'LC_LOAD_DYLIB' and line[0] == 'name':
            lib = line[1].rsplit(None, 2)[0]
            if (lib.startswith(OVERRIDE_LIBS) or
                    not lib.startswith(SYSTEM_LIBDIRS)):
                print('warning:', filename, 'check:', lib)
            cmd = None
        elif cmd == 'LC_RPATH' and line[0] == 'path':
            path = line[1].rsplit(None, 2)[0]
            if (path.startswith(OVERRIDE_LIBS) or
                    not path.startswith(SYSTEM_LIBDIRS)):
                print('warning:', filename, 'rpath:', path)
            cmd = None


def scan_dir(start_dir):
    if start_dir.endswith('/'):
        start_dir = start_dir[:-1]
    if start_dir.endswith('app'):
        start_dir += '/Contents'
        lib_dir = '%s/lib' % start_dir
        prog_dirs = ['%s/bin' % start_dir, '%s/MacOS' % start_dir]
    else:
        lib_dir = '%s/lib' % start_dir
        prog_dirs = ['%s/bin' % start_dir]
    for dirpath, dirnames, filenames in os.walk(start_dir):
        if dirpath == 'build':
            dirnames.remove('tmp')
        filenames = [
            fn for fn in filenames if not fn.endswith(IGNORE_SUFFIXES)
        ]
        if not filenames:
            continue
        if dirpath in prog_dirs:
            for fn in filenames:
                check_otool(os.path.join(dirpath, fn), program=True)
            continue
        # normal check
        for fn in filenames:
            check_otool(os.path.join(dirpath, fn), rpath=(dirpath == lib_dir))


scan_dir(directory)
raise SystemExit(0)
