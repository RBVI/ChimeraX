#!/usr/bin/env python3
# vi: set sw=4 et:
"""
How to take a ChimeraX.app on Linux and turn it into a debian package
for Ubuntu

References:
https://wiki.debian.org/FilesystemHierarchyStandard
https://www.debian.org/doc/debian-policy/
https://www.freedesktop.org/wiki/Specifications/desktop-entry-spec/
#
Designed for Ubuntu, would mostly work on Debian except that ffmpeg
not available in Debian

Steps:
  1. build ChimeraX without Debian supplied prerequisites
  2. create Debian binary package layout
  3. build Deiban package
"""

import os
import subprocess
import sys
import tempfile
import textwrap

# app_author and app_name are the same as ChimeraX_main.py
app_author = "UCSF"
app_name = "ChimeraX"

CHIMERAX_INSTALL = f"{app_name}.app"
CHIMERAX_BIN = f"{CHIMERAX_INSTALL}/bin/{app_name}"

# lintian(1) complains about files /opt
# INST_DIR = "opt"
INST_DIR = "usr/lib"

UBUNTU_DEPENDENCIES = {
    "16.04": {
        "libasound2": "1.1.0",
        "libatk1.0-0": "2.18.0",
        "libbz2-1.0": "1.0.6",
        "libc6": "2.23",
        "libcairo-gobject2": "1.14.6",
        "libcairo2": "1.14.6",
        "libcups2": "2.1.3",
        "libdbus-1-3": "1.10.6",
        "libdrm2": "2.4.83",
        "libegl1-mesa": "17.2.8",
        "libexpat1": "2.1.0",
        "libfftw3-single3": "3.3.4",
        "libfontconfig1": "2.11.94",
        "libfreetype6": "2.6.1",
        "libgcc1": "6.0.1",
        "libgdk-pixbuf2.0-0": "2.32.2",
        "libgfortran3": "5.4.0",
        "libgl1-mesa-glx": "17.2.8",
        "libglib2.0-0": "2.48.2",
        "libglu1-mesa": "9.0.0",
        "libgstreamer-plugins-base1.0-0": "1.8.3",
        "libgstreamer1.0-0": "1.8.3",
        "libgtk-3-0": "3.18.9",
        "libllvm3.8": "3.8",
        "liblzma5": "5.1.1alpha+20120614",
        "libnspr4": "4.13.1",
        "libnss3": "3.28.4",
        "libpango-1.0-0": "1.38.1",
        "libpangocairo-1.0-0": "1.38.1",
        "libpulse-mainloop-glib0": "8.0",
        "libpulse0": "8.0",
        "libsqlite3-0": "3.11.0",
        "libssl1.0.0": "1.0.2g",
        "libstdc++6": "5.4.0",
        "libx11-6": "1.6.3",
        "libx11-xcb1": "1.6.3",
        "libxcb-glx0": "1.11.1",
        "libxcb-util1": "0.4.0",
        "libxcb-xfixes0": "1.11.1",
        "libxcb-xinerama0": "1.11.1",
        "libxcb1": "1.11.1",
        "libxcomposite1": "0.4.4",
        "libxcursor1": "1.1.14",
        "libxdamage1": "1.1.4",
        "libxext6": "1.3.3",
        "libxfixes3": "5.0.1",
        "libxi6": "1.7.6",
        "libxrandr2": "1.5.0",
        "libxrender1": "0.9.9",
        "libxtst6": "1.2.2",
        "zlib1g": "1.2.8.dfsg",
        "xdg-utils": "1.0",  # TODO: get right version
    },
    "18.04": {
        "libasound2": "1.1.3",
        "libatk1.0-0": "2.28.1",
        "libbz2-1.0": "1.0.6",
        "libc6": "2.27",
        "libcairo-gobject2": "1.15.10",
        "libcairo2": "1.15.10",
        "libcups2": "2.2.7",
        "libdbus-1-3": "1.12.2",
        "libdrm2": "2.4.91",
        "libegl1": "1.0.0",
        "libexpat1": "2.2.5",
        "libffi6": "3.2.1",
        "libfftw3-single3": "3.3.7",
        "libfontconfig1": "2.12.6",
        "libfreetype6": "2.8.1",
        "libgcc1": "8-20180414",
        "libgdk-pixbuf2.0-0": "2.36.11",
        "libgfortran4": "7.5.0",
        "libgl1": "1.0.0",
        "libglib2.0-0": "2.56.1",
        "libglu1-mesa": "9.0.0",
        "libgstreamer-plugins-base1.0-0": "1.14.1",
        "libgstreamer1.0-0": "1.14.1",
        "libgtk-3-0": "3.22.30",
        "liblzma5": "5.2.2",
        "libnspr4": "4.18",
        "libnss3": "3.35",
        "libosmesa6": "18.0.0",
        "libpango-1.0-0": "1.40.14",
        "libpangocairo-1.0-0": "1.40.14",
        "libpulse-mainloop-glib0": "11.1",
        "libpulse0": "11.1",
        "libsqlite3-0": "3.22.0",
        "libssl1.1": "1.1.0g",
        "libstdc++6": "8-20180414",
        "libx11-6": "1.6.4",
        "libx11-xcb1": "1.6.4",
        "libxcb-glx0": "1.13",
        "libxcb-util1": "0.4.0",
        "libxcb-xfixes0": "1.13",
        "libxcb-xinerama0": "1.13",
        "libxcb1": "1.13",
        "libxcomposite1": "0.4.4",
        "libxcursor1": "1.1.15",
        "libxdamage1": "1.1.4",
        "libxext6": "1.3.3",
        "libxfixes3": "5.0.3",
        "libxi6": "1.7.9",
        "libxrandr2": "1.5.1",
        "libxrender1": "0.9.10",
        "libxtst6": "1.2.3",
        "xdg-utils": "1.1.2",
        "zlib1g": "1.2.11.dfsg",
    },
    "20.04": {
        "libasound2": "1.2.2",
        "libatk1.0-0": "2.35.1",
        "libbz2-1.0": "1.0.8",
        "libc6": "2.31",
        "libcairo-gobject2": "1.16.0",
        "libcairo2": "1.16.0",
        "libcrypt1": "4.4.10",
        "libcups2": "2.3.1",
        "libdbus-1-3": "1.12.16",
        "libdrm2": "2.4.101",
        "libegl1": "1.3.1",
        "libexpat1": "2.2.9",
        "libffi7": "3.3",
        "libfftw3-single3": "3.3.8",
        "libfontconfig1": "2.13.1",
        "libfreetype6": "2.10.1",
        "libgcc-s1": "10-20200411",
        "libgdk-pixbuf2.0-0": "2.40.0+dfsg",
        "libgfortran5": "10-20200411",
        "libgl1": "1.3.1",
        "libglib2.0-0": "2.64.2",
        "libglu1-mesa": "9.0.1",
        "libgstreamer-plugins-base1.0-0": "1.16.2",
        "libgstreamer1.0-0": "1.16.2",
        "libgtk-3-0": "3.24.18",
        "liblzma5": "5.2.4",
        "libncursesw6": "6.2",
        "libnspr4": "4.25",
        "libnss3": "3.49.1",
        "libopenjp2-7": "2.3.1",
        "libosmesa6": "20.0.4",
        "libpango-1.0-0": "1.44.7",
        "libpangocairo-1.0-0": "1.44.7",
        "libpulse-mainloop-glib0": "13.99.1",
        "libpulse0": "13.99.1",
        "libsqlite3-0": "3.31.1",
        "libssl1.1": "1.1.1f",
        "libstdc++6": "10-20200411",
        "libtinfo6": "6.2",
        "libuuid1": "2.34",
        "libwayland-client0": "1.18.0",
        "libwayland-cursor0": "1.18.0",
        "libwayland-egl1": "1.18.0",
        "libwayland-server0": "1.18.0",
        "libx11-6": "1.6.9",
        "libx11-xcb1": "1.6.9",
        "libxcb-glx0": "1.14",
        "libxcb-util1": "0.4.0",
        "libxcb-xinerama0": "1.14",
        "libxcb1": "1.14",
        "libxcomposite1": "0.4.5",
        "libxcursor1": "1.2.0",
        "libxdamage1": "1.1.5",
        "libxext6": "1.3.4",
        "libxfixes3": "5.0.3",
        "libxi6": "1.7.10",
        "libxkbcommon0": "0.10.0",
        "libxrender1": "0.9.10",
        "libxtst6": "1.2.3",
        "xdg-utils": "1.1.3",
        "zlib1g": "1.2.11.dfsg",
    },
    "22.04": {
        # TODO: update versions
        "libasound2": "1.2.6.1",
        "libatk1.0-0": "2.36.0",
        "libbz2-1.0": "1.0.8",
        "libc6": "2.35",
        "libcairo-gobject2": "1.16.0",
        "libcairo2": "1.16.0",
        "libcrypt1": "1:4.4.17",
        "libcups2": "2.3.1",
        "libdbus-1-3": "1.12.16",
        "libdrm2": "2.4.101",
        "libegl1": "1.3.1",
        "libexpat1": "2.2.9",
        "libffi7": "3.3",
        "libfftw3-single3": "3.3.8",
        "libfontconfig1": "2.13.1",
        "libfreetype6": "2.10.1",
        "libgcc-s1": "10-20200411",
        "libgdk-pixbuf2.0-0": "2.40.0+dfsg",
        "libgfortran5": "10-20200411",
        "libgl1": "1.3.1",
        "libglib2.0-0": "2.64.2",
        "libglu1-mesa": "9.0.1",
        "libgstreamer-plugins-base1.0-0": "1.16.2",
        "libgstreamer1.0-0": "1.16.2",
        "libgtk-3-0": "3.24.18",
        "liblzma5": "5.2.4",
        "libncursesw6": "6.2",
        "libnspr4": "4.25",
        "libnss3": "3.49.1",
        "libopenjp2-7": "2.3.1",
        "libosmesa6": "20.0.4",
        "libpango-1.0-0": "1.44.7",
        "libpangocairo-1.0-0": "1.44.7",
        "libpulse-mainloop-glib0": "13.99.1",
        "libpulse0": "13.99.1",
        "libsqlite3-0": "3.31.1",
        "libssl3": "3.0.2",
        "libstdc++6": "10-20200411",
        "libtinfo6": "6.2",
        "libuuid1": "2.34",
        "libwayland-client0": "1.18.0",
        "libwayland-cursor0": "1.18.0",
        "libwayland-egl1": "1.18.0",
        "libwayland-server0": "1.18.0",
        "libx11-6": "1.6.9",
        "libx11-xcb1": "1.6.9",
        "libxcb-glx0": "1.14",
        "libxcb-util1": "0.4.0",
        "libxcb-xinerama0": "1.14",
        "libxcb1": "1.14",
        "libxcomposite1": "0.4.5",
        "libxcursor1": "1.2.0",
        "libxdamage1": "1.1.5",
        "libxext6": "1.3.4",
        "libxfixes3": "5.0.3",
        "libxi6": "1.7.10",
        "libxkbcommon0": "0.10.0",
        "libxrender1": "0.9.10",
        "libxtst6": "1.2.3",
        "xdg-utils": "1.1.3",
        "zlib1g": "1.2.11.dfsg",
    },
}


def main():
    """main program"""
    # assume ChimeraX.app already has appropriate binaries
    error = False
    os_version = None
    build = 'release'
    if len(sys.argv) >= 2:
        os_version = sys.argv[1]
    else:
        error = True
    if len(sys.argv) >= 3:
        build = sys.argv[2]
        if build not in ['release', 'candidate', 'daily', 'techpreview']:
            error = True
    if error or len(sys.argv) > 3 or os_version not in UBUNTU_DEPENDENCIES:
        print(f'Usage: {sys.argv[0]} ubuntu-version [build-type]', file=sys.stderr)
        print('  Supported Ubuntu versions are:', ', '.join(UBUNTU_DEPENDENCIES.keys()),
              file=sys.stderr)
        print('  Build-type is one of "release", "candidate", or "daily"', file=sys.stderr)
        raise SystemExit(2)
    dependencies = UBUNTU_DEPENDENCIES[os_version]
    output = subprocess.check_output([
        CHIMERAX_BIN, "--nocolor", "--version"], stderr=subprocess.DEVNULL).decode()
    full_version = [line for line in output.split('\n') if 'version:' in line]
    if not full_version:
        print("Not able to determine version number")
        raise SystemExit(1)
    full_version = full_version[0].split(':', maxsplit=1)[1].strip()
    version_number, version_date = full_version.split(maxsplit=1)
    from packaging.version import Version
    version = Version(version_number)
    version_date = version_date[1:-1].replace('-', '.')
    pkg_name = f"{app_author.lower()}-{app_name.lower()}"
    bin_name = app_name.lower()  # name of command in /usr/bin
    if build == 'daily':
        # daily build, version is date
        version = version_date
        pkg_name += "-daily"
        bin_name += "-daily"
    elif build == 'techpreview':
        # like daily build, version is date
        version = version_date
        pkg_name += "-techpreview"
        bin_name += "-techpreview"
    elif build == 'release':
        # release build
        version = version.base_version
    else:
        # candidate build
        version = f"{version.base_version}+rc{version_date}"
    version = f"{version}ubuntu{os_version}"
    deb_name = f"{pkg_name}-{version}"  # name of .deb file

    # print('full_version:', repr(full_version))
    # print('version_number:', version_number)
    # print('version_date:', version_date)
    # print('deb_name:', deb_name)
    # print('pkg_name:', pkg_name)
    # print('bin_name:', bin_name)

    os.umask(0o22)  # turn off group and other writability
    pkg_root = f"{deb_name}"
    os.mkdir(pkg_root)
    debian_dir = f"{pkg_root}/DEBIAN"
    os.mkdir(debian_dir)
    make_control_file(debian_dir, pkg_name, version, dependencies)
    make_postinst(debian_dir, pkg_name, bin_name)
    make_prerm(debian_dir, pkg_name, bin_name)
    doc_dir = f"{pkg_root}/usr/share/doc/{pkg_name}"
    os.makedirs(doc_dir)
    make_copyright_file(doc_dir)
    make_changelog_file(doc_dir)
    # add_readme(doc_dir)

    copy_app(pkg_root, pkg_name)
    make_bin(pkg_root, pkg_name, bin_name)
    src = f'{app_name}.app/share/man/man1/{app_name}.1'
    make_man_file(src, pkg_root, bin_name)

    subprocess.check_call(['/bin/chmod', '-R', 'a+rX,go-w', pkg_root])

    dst_dir = f'./ubuntu-{os_version}'
    os.makedirs(dst_dir, exist_ok=True)

    # fakeroot environment file
    frenv = tempfile.NamedTemporaryFile(prefix='frenv-', suffix='.db')
    subprocess.check_call([
        '/usr/bin/fakeroot',
        '-s', frenv.name,
        '--',
        '/bin/chown',
        '-hRP',
        '0.0',
        pkg_root
    ])
    subprocess.check_call([
        '/usr/bin/fakeroot',
        '-i', frenv.name,
        '--',
        '/usr/bin/dpkg-deb',
        '--build',
        pkg_root,
        dst_dir
    ])


def copy_app(pkg_root, pkg_name):
    """Copy application to correct place in package hierarchy"""
    import shutil
    src = f'{app_name}.app'
    dst_dir = f'{pkg_root}/{INST_DIR}'
    dst = f'{pkg_root}/{INST_DIR}/{pkg_name}'
    os.makedirs(dst_dir)
    shutil.copytree(src, dst)

    # cleanup -- remove __pycache__ directories
    cache_dirs = subprocess.check_output([
        '/usr/bin/find',
        dst,
        '-name', '__pycache__'
        ]).strip().decode().split('\n')
    for d in cache_dirs:
        shutil.rmtree(d)

    # cleanup -- remove python shell scripts
    for fn in os.listdir(f'{dst}/bin'):
        filename = f'{dst}/bin/{fn}'
        if not os.path.isfile(filename):
            continue
        with open(filename, 'rb') as f:
            header = f.read(64)
        if header[0:2] != b'#!':
            continue
        program = header[2:].lstrip()
        if program.startswith(b'/bin/') or \
                program.startswith(b'/usr/bin/'):
            continue
        os.remove(filename)


def make_bin(pkg_root, pkg_name, bin_name):
    """Make symbolic link to executable"""
    bin_dir = f'{pkg_root}/usr/bin'
    os.makedirs(bin_dir, exist_ok=True)
    os.symlink(f'../../{INST_DIR}/{pkg_name}/bin/{app_name}', f'{bin_dir}/{bin_name}')


def make_control_file(debian_dir, pkg_name, version, dependencies):
    """Make control file"""
    # see "man 1 deb-control"
    if dependencies is not None:
        deps = list(dependencies)
    else:
        deps = []
    deps.insert(0, 'ffmpeg')
    depends = ', '.join(deps)
    with open(f"{debian_dir}/control", 'w') as f:
        print(textwrap.dedent(f"""\
            Package: {pkg_name}
            Version: {version}
            Architecture: amd64
            Depends: {depends}
            Maintainer: Chimera Staff <chimera-staff@cgl.ucsf.edu>
            Description: molecular visualization
             UCSF ChimeraX (or simply ChimeraX) is the next-generation
             molecular visualization program from the Resource for Biocomputing
             Visualization, and Informatics (RBVI), following UCSF Chimera.
             ChimeraX can be downloaded free of charge for academic, government
             nonprofit, and personal use. Commercial users, please see licensing.
            Homepage: https://www.rbvi.ucsf.edu/chimerax/
            Bugs: mailto:chimerax-bugs@cgl.ucsf.edu
            Section: contrib/science
            Priority: optional
            Suggests: opencl-icd
            Tag: science::visualisation, science::modelling, field::biology, field::chemistry,
             field::biology:structural, field::biology:bioinformatics,
             biology::nucleic-acids, biology::peptidic,
             use::viewing, use::analysing,
             scope::application, x11::application, role::program,
             interface::3d, interface::graphical, interface::commandline,
             implemented-in::c++, implemented-in::python, uitoolkit::qt,
             network::client
            """), file=f)


def make_postinst(debian_dir, pkg_name, bin_name):
    # see "man 1 deb-postinst"
    with open(f"{debian_dir}/postinst", 'w') as f:
        print(textwrap.dedent(f"""\
            #!/bin/sh
            set -e
            case "$1" in
                configure)
                    echo "Install desktop menu and associated mime types"
                    {bin_name} --exit --nogui --silent --cmd 'linux xdg-install system true'
                    echo "Precompiling Python packages"
                    ({bin_name} -m compileall /{INST_DIR}/{pkg_name} || exit 0)
                ;;
            esac
            """), file=f)
        os.fchmod(f.fileno(), 0o755)


def make_prerm(debian_dir, pkg_name, bin_name):
    # see "man 1 deb-prerm"
    with open(f"{debian_dir}/prerm", 'w') as f:
        print(textwrap.dedent(f"""\
            #!/bin/sh
            set -e
            echo "Deregister desktop menu and associated mime types"
            {bin_name} --exit --nogui --silent --cmd 'linux xdg-uninstall system true'
            echo "Remove Python cache files"
            find /{INST_DIR}/{pkg_name} -name __pycache__ -print0 | xargs -0 /bin/rm -rf
            """), file=f)
        os.fchmod(f.fileno(), 0o755)


def make_copyright_file(doc_dir):
    """Copy the copyright file"""
    # copyright file
    with open("copyright.txt") as f:
        our_copyright = f.read()
    with open(f"{doc_dir}/copyright", 'w') as f:
        print(textwrap.dedent("""\
            Format: https://www.debian.org/doc/packaging-manuals/copyright-format/1.0/

            Upstream-Name: ChimeraX
            Upstream-Contact: chimerax@rbvi.ucsf.edu
            Source: https://www.rbvi.ucsf.edu/chimerax/

            Disclaimer:  Package is maintained by the ChimeraX team
              Copyrights for embedded code are given in the documentation
              in INSTALLDIR/share/docs/embeded.html

            Comment:

              The computer code and documentation that comprises UCSF ChimeraX is protected
              by copyrights held by The Regents of the University of California ("The Regents")
              and is licensed for non-commercial use at no cost by The Regents.

              COMMERCIAL USE OF UCSF CHIMERAX IS COVERED BY A SEPARATE WRITTEN LICENSE AGREEMENT.
              Commercial licensing costs are tier-based, depending on the number of users.
              Please email chimerax@cgl.ucsf.edu
              if you are interested in using ChimeraX for commercial purposes.

            License: free for non-commercial use
            """), file=f)
        for line in our_copyright.split('\n'):
            if line.startswith("=== UCSF ChimeraX"):
                continue
            print(' ', line, file=f)


def gzip(filename):
    """Gzip without saving timestamp (nor name) so it is reproducible"""
    subprocess.check_call(['gzip', '-n9', filename])


def make_changelog_file(doc_dir):
    """Copy the change log file"""
    filename = f"{doc_dir}/changelog"
    with open(filename, 'wt') as f:
        print("See https://www.rbvi.ucsf.edu/trac/ChimeraX/wiki/ChangeLog", file=f)
    gzip(filename)


def make_man_file(src_file, pkg_root, bin_name):
    base, ext = os.path.splitext(src_file)
    if not ext or not os.path.exists(src_file):
        return
    section = ext[1:]
    man_dir = f'{pkg_root}/usr/share/man/man{section}'
    man_file = f'{man_dir}/{bin_name}.{section}'
    os.makedirs(man_dir, exist_ok=True)

    with open(src_file, 'rb') as f:
        src = f.read()
    with open(man_file, 'wb') as f:
        f.write(src)
    gzip(man_file)


if __name__ == '__main__':
    main()
