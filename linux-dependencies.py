#!/usr/bin/env python3
# vi: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2018 Regents of the University of California.
# All rights reserved.  This software provided pursuant to a
# license agreement containing restrictions on its disclosure,
# duplication and use.  For details see:
# https://www.rbvi.ucsf.edu/chimerax/docs/licensing.html
# This notice must be embedded in or attached to all copies,
# including partial copies, of the software or any revisions
# or derivations thereof.
# === UCSF ChimeraX Copyright ===

#
# On Linux, scan build tree for shared libraries and executables
# and figure out the packages needed to support them.

import os
import subprocess
import sys

# speed up processing by ignoring files with the following
IGNORE_SUFFIXES = set([
    '.a', '.o', '.py', '.pyc', '.pyw', '.cpp', '.h', '.pyx', '.c', '.f', '.f90', '.pth', '.i',
    '.cif', '.pdb', '.stl', '.pdb', '.mmcif', '.mol2', '.cxc', '.h5', '.bild',
    '.rst', '.html', '.css', '.css_t', '.js', '.js_t', '.json', '.xml', '.xsl',
    '.png', '.svg', '.jpg', '.gif',
    '.txt', '.idatmres',
    '.qml', '.qmlc', '.dat', '.qm', '.pak', '.qmltypes', '.mplstyle',
    '.mat', '.sav',
    '.gz', '.bz2', '.xz',
    '.diag', '.matrix', '.po', '.mo',
    '.afm', '.pdf', '.ttf', '.mp4', '.ogv', '.wav',
    '.conf', '.cfg', '.def', '.rc',
    '.npy', '.npz', '.arff',
])

# % ldd cxsrc/ChimeraX.app/lib/libelement.so
#        linux-vdso.so.1 (0x00007fffb3565000)
#        libpython3.6m.so.1.0 => /usr/lib/x86_64-linux-gnu/libpython3.6m.so.1.0 (0x00007fdafd6e4000)
#        libpyinstance.so => not found
#        libstdc++.so.6 => /usr/lib/x86_64-linux-gnu/libstdc++.so.6 (0x00007fdafd356000)
#        libgcc_s.so.1 => /lib/x86_64-linux-gnu/libgcc_s.so.1 (0x00007fdafd13e000)
#        libc.so.6 => /lib/x86_64-linux-gnu/libc.so.6 (0x00007fdafcd4d000)
#        libexpat.so.1 => /lib/x86_64-linux-gnu/libexpat.so.1 (0x00007fdafcb1b000)
#        libz.so.1 => /lib/x86_64-linux-gnu/libz.so.1 (0x00007fdafc8fe000)
#        libpthread.so.0 => /lib/x86_64-linux-gnu/libpthread.so.0 (0x00007fdafc6df000)
#        libdl.so.2 => /lib/x86_64-linux-gnu/libdl.so.2 (0x00007fdafc4db000)
#        libutil.so.1 => /lib/x86_64-linux-gnu/libutil.so.1 (0x00007fdafc2d8000)
#        libm.so.6 => /lib/x86_64-linux-gnu/libm.so.6 (0x00007fdafbf3a000)
#        /lib64/ld-linux-x86-64.so.2 (0x00007fdafdda8000)
# % dpkg -S /lib/x86_64-linux-gnu/libz.so.1
# zlib1g:amd64: /lib/x86_64-linux-gnu/libz.so.1
# % readelf -d cxsrc/ChimeraX.app/bin/ChimeraX | grep NEED
# 0x0000000000000001 (NEEDED)             Shared library: [libpython3.6m.so.1.0]
# 0x0000000000000001 (NEEDED)             Shared library: [libc.so.6]


libraries = {}
not_found = []
seen = set()


def get_dependencies(filename, pkg_type, env=False, start_dir=None):
    needed = set()
    # get list of libraries binary is actually linked against
    try:
        output = subprocess.check_output(
            ['/usr/bin/readelf', '-d', filename], stderr=subprocess.DEVNULL, encoding='utf-8')
    except subprocess.CalledProcessError:
        return
    lines = [x.strip() for x in output.split('\n')]
    for line in lines:
        tokens = line.split()
        if len(tokens) < 5 or tokens[1] != '(NEEDED)':
            continue
        lib = tokens[4][1:-1]
        needed.add(lib)
    # see which libraries are used
    try:
        output = subprocess.check_output(
            ['/usr/bin/ldd', filename], env=env, stderr=subprocess.DEVNULL, encoding='utf-8')
    except subprocess.CalledProcessError:
        return
    libname = os.path.split(filename)[1]
    print(f'working on {repr(libname)}', file=sys.stderr)  # DEBUG
    seen.add(libname)
    lines = [x.strip() for x in output.split('\n')]
    for line in lines:
        tokens = line.split()
        if len(tokens) < 3 or tokens[1] != '=>' or tokens[0] not in needed:
            continue
        lib = tokens[2]
        if lib == 'not':
            not_found.append((tokens[0], filename))
            continue
        if lib in libraries:
            # skip libraries we've already seen
            continue
        if start_dir is not None and lib.startswith(start_dir):
            # skip internal libraries
            continue
        pkg = get_package_for_lib(lib, pkg_type)
        if pkg is not None:
            libraries[lib] = pkg


def scan_dir(start_dir, pkg_type):
    # look for shared libraries, error on the side of checking too many files
    if start_dir.endswith('/'):
        start_dir = start_dir[:-1]
    env = {
        #'LD_LIBRARY_PATH': f'{start_dir}/lib:{start_dir}/lib/python3.7/site-packages/PyQt5'
        'LD_LIBRARY_PATH': f'{start_dir}/lib'
    }
    for dirpath, dirnames, filenames in os.walk(start_dir):
        if dirpath == 'build':
            dirnames.remove('tmp')
        filenames = [
            fn for fn in filenames if os.path.splitext(fn)[1] not in IGNORE_SUFFIXES
        ]
        for fn in filenames:
            get_dependencies(os.path.join(dirpath, fn), pkg_type, env=env)


def extract_version(version):
    # "The format is: [epoch:]upstream_version[-debian_revision]."
    if ':' in version:
        version = version.split(':', 1)[1]
    if '-' in version:
        version = version.rsplit('-', 1)[0]
    return version


def get_package_versions(packages, pkg_type):
    if pkg_type == 'deb':
        try:
            output = subprocess.check_output(
                ['/usr/bin/dpkg-query', '--show', '--showformat=${Package} ${Version}\\n']
                + list(packages), stderr=subprocess.DEVNULL, encoding='utf-8')
            pkg_info = output.split()
            return dict(zip(pkg_info[0::2], pkg_info[1::2]))
        except subprocess.CalledProcessError:
            pass
    if pkg_type == 'rpm':
        # packages already include version
        return dict([p.rsplit('-', 2)[0:2] for p in packages])
    raise RuntimeError('{pkg_type} is not supported')


def get_package_for_lib(lib, pkg_type):
    if pkg_type == 'deb':
        try:
            output = subprocess.check_output(
                ['/usr/bin/dpkg', '-S', lib], stderr=subprocess.DEVNULL, encoding='utf-8')
        except subprocess.CalledProcessError:
            lib = os.path.realpath(lib)
            try:
                output = subprocess.check_output(
                    ['/usr/bin/dpkg', '-S', lib], stderr=subprocess.DEVNULL, encoding='utf-8')
            except subprocess.CalledProcessError:
                return
        output = output.strip()
        return output.split(None, 1)[0].split(':', 1)[0]
    if pkg_type == 'rpm':
        # actually returns "package-version-build"
        try:
            output = subprocess.check_output(
                ['/usr/bin/rpm', '-q', '--whatprovides', lib], stderr=subprocess.DEVNULL, encoding='utf-8')
        except subprocess.CalledProcessError:
            return
        return output.strip()
    raise RuntimeError('{pkg_type} is not supported')


def packages_needed_by(packages, pkg_type):
    # Read package dependency information.
    # If a depends on b, then b is needed
    # by a (and doesn't need to be a dependency).
    # Return { pkg: [ pkg needed by ] }
    needed_by = {}
    if pkg_type == 'rpm':
        for pkg in packages:
            try:
                output = subprocess.check_output(
                    ['/usr/bin/rpm', '-qR', pkg], stderr=subprocess.DEVNULL, encoding='utf-8')
            except subprocess.CalledProcessError:
                continue
            for line in output.split('\n'):
                line = line.strip()
                if not line or line.startswith('/') or ' ' in line or '(' in line:
                    continue
                needed = needed_by.setdefault(line, [])
                needed.append(pkg)
    elif pkg_type == 'deb':
        for pkg in packages:
            try:
                output = subprocess.check_output(
                    ['/usr/bin/apt-cache', 'depends', pkg],
                    stderr=subprocess.DEVNULL, encoding='utf-8')
            except subprocess.CalledProcessError:
                continue
            for line in output.split('\n'):
                line = line.strip()
                if not line.startswith('Depends:'):
                    continue
                other = line.split()[-1]
                needed = needed_by.setdefault(other, [])
                needed.append(pkg)
    return needed_by


def main(directory, pkg_type):
    scan_dir(directory, pkg_type)
    # pretend we saw CUDA libraries
    seen.update(['libcuda.so.1', 'libcufft.so.9.0', 'libnvrtc.so.9.0'])
    seen.update(['libcufft.so.10', 'libnvrtc.so.10'])
    # pretend we saw OpenCL libraries
    seen.update(['libOpenCL.so.1'])

    # don't provide packages for libraries we provide
    packages = set([pkg for lib, pkg in libraries.items() if os.path.split(lib)[1] not in seen])
    if not packages:
        print('No packages needed')
    else:
        import glob
        if pkg_type == 'deb':
            osmesas = glob.glob("/usr/lib/x86_64-linux-gnu/libOSMesa.so*")
            if osmesas:
                pkg = get_package_for_lib(osmesas[0], "deb")
                if pkg is not None:
                    packages.add(pkg)
            if 'xdg-utils' not in packages:
                packages.add('xdg-utils')
            # don't depend on Postgres
            packages.discard('libpq5')
            # don't depend on Qt multimedia gstreamer tools
            packages.discard('libqgsttools-p1')
        elif pkg_type == 'rpm':
            osmesas = glob.glob("/usr/lib64/libOSMesa.so*")
            if osmesas:
                osmesas.sort(key=len, reverse=True)
                pkg = get_package_for_lib(osmesas[0], "rpm")
                if pkg is not None:
                    packages.add(pkg)
            pkg = get_package_for_lib("/usr/bin/xdg-desktop-menu", "rpm")
            if pkg is None:
                packages.add("xdg-utils-1.1.0-el7.noarch")
            else:
                packages.add(pkg)
        package_versions = get_package_versions(packages, pkg_type)
        packages = list(package_versions.keys())
        if pkg_type == 'rpm':
            # easier to postprocess to remove unwanted libraries
            for p in packages:
                if p.startswith('postgres'):
                    del package_versions[p]
            packages = list(package_versions.keys())
        packages.sort(key=str.casefold)
    #
    print('Packages needed:')
    skipped = []
    needed_by = packages_needed_by(packages, pkg_type)
    for name in sorted(packages):
        if pkg_type != 'deb':
            # debian has circular dependencies that messes this up
            if name in needed_by and any(pkg in packages for pkg in needed_by[name]):
                skipped.append(name)
                continue
        ver = extract_version(package_versions[name])
        print(f'   "{name}": "{ver}",')
    #
    if skipped:
        print("Skipped:")
        for pkg in sorted(skipped):
            print(f'   "{pkg}": "{needed_by[pkg]}",')
    #
    missing = [n for n in not_found if n[0] not in seen]
    if missing:
        missing.sort()
        print()
        print("Not found:")
        for n in missing:
            print('   %s not found in %s' % n)
    if 1:
        # show raw library data
        print()
        libs = list(libraries.keys())
        libs.sort()
        for lib in libs:
            package = libraries[lib]
            print(f'"{lib}": "{package}"')
    raise SystemExit(0)


if __name__ == '__main__':
    if not sys.platform.startswith('linux'):
        print('only works on Linux', file=sys.stderr)
        raise SystemExit(1)
    import getopt
    directory = None
    pkg_type = 'deb'
    if len(sys.argv) >= 2:
        directory = sys.argv[1]
    if len(sys.argv) >= 3:
        pkg_type = sys.argv[2]
    if directory is None or len(sys.argv) > 3:
        print('usage: %s directory [package-type]' % sys.argv[0], file=sys.stderr)
        print('  package-type is one of "deb" or "rpm"', file=sys.stderr)
        raise SystemExit(2)
    main(directory, pkg_type)
