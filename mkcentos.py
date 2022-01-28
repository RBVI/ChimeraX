#!/usr/bin/env python3
# vi: set sw=4 et:
"""
How to take a ChimeraX.app on Linux and turn it into a rpm package
for Centos

Expects ChimeraX.app to be in the current directory.
Uses ~/rpmbuild -- change HOME environment variable if need be.

References:
http://rpm-guide.readthedocs.io/en/latest/rpm-guide.html
http://rpm.org/documentation.html
http://ftp.rpm.org/max-rpm/ch-rpm-inside.html
https://stackoverflow.com/questions/880227/what-is-the-minimum-i-have-to-do-to-create-an-rpm-file
#
Designed for Centos, should work on RedHat and Fedora

Steps:
  1. build ChimeraX
  2. create rpm binary package layout
  3. build rpm package
"""

import os
import subprocess
import sys
import textwrap

# app_author and app_name are the same as ChimeraX_main.py
app_author = "UCSF"
app_name = "ChimeraX"

CHIMERAX_INSTALL = f"{os.getcwd()}/{app_name}.app"
CHIMERAX_BIN = f"{CHIMERAX_INSTALL}/bin/{app_name}"

PREFIX = "/usr"
if PREFIX == "/opt":
    APP_DIR = f"{app_author}/{app_name}"
else:
    APP_DIR = f"libexec/{app_author}-{app_name}"

CENTOS_DEPENDENCIES = {
    "7": {
       "alsa-lib": "1.1.8",
       "atk": "2.28.1",
       "cairo": "1.15.12",
       "cairo-gobject": "1.15.12",
       "cups-libs": "1.6.3",
       "dbus-libs": "1.10.24",
       "expat": "2.1.0",
       "fftw-libs-single": "3.3.3",
       "fontconfig": "2.13.0",
       "freetype": "2.8",
       "gdk-pixbuf2": "2.36.12",
       "glib2": "2.56.1",
       "glibc": "2.17",
       "gstreamer1": "1.10.4",
       "gstreamer1-plugins-base": "1.10.4",
       "gtk3": "3.22.30",
       "krb5-libs": "1.15.1",
       "libdrm": "2.4.97",
       "libffi": "3.0.13",
       "libgcc": "4.8.5",
       "libgfortran4": "4.8.5",
       "libglvnd-egl": "1.0.1",
       "libglvnd-glx": "1.0.1",
       "libstdc++": "4.8.5",
       "libuuid": "2.23.2",
       "libwayland-client": "1.15.0",
       "libwayland-cursor": "1.15.0",
       "libwayland-egl": "1.15.0",
       "libX11": "1.6.7",
       "libxcb": "1.13",
       "libXcomposite": "0.4.4",
       "libXcursor": "1.1.15",
       "libXdamage": "1.1.4",
       "libXext": "1.3.3",
       "libXfixes": "5.0.3",
       "libXi": "1.7.9",
       "libxkbcommon": "0.7.1",
       "libxkbcommon-x11": "0.7.1",
       "libXrandr": "1.5.1",
       "libXrender": "0.9.10",
       "libXtst": "1.2.3",
       "mesa-libGLU": "9.0.0",
       "mesa-private-llvm": "3.9.1",
       "nspr": "4.21.0",
       "nss": "3.44.0",
       "nss-util": "3.44.0",
       "openjpeg2": "2.3.1",
       "openssl-libs": "1.0.2k",
       "pango": "1.42.4",
       "pulseaudio-libs": "10.0",
       "pulseaudio-libs-glib2": "10.0",
       "sqlite": "3.7.17",
       "xcb-util-keysyms": "0.4.0",
       "xdg-utils": "1.1.0",
       "xz-libs": "5.2.2",
       "zlib": "1.2.7",
    },
    "8": {
       "alsa-lib": "1.1.9",
       "atk": "2.28.1",
       "cairo": "1.15.12",
       "cairo-gobject": "1.15.12",
       "cups-libs": "2.2.6",
       "dbus-libs": "1.12.8",
       "expat": "2.2.5",
       "fftw-libs-single": "3.3.5",
       "fontconfig": "2.13.1",
       "freetype": "2.9.1",
       "gdk-pixbuf2": "2.36.12",
       "glib2": "2.56.4",
       "glibc": "2.28",
       "gtk3": "3.22.30",
       "libdrm": "2.4.98",
       "libffi": "3.1",
       "libgcc": "8.3.1",
       "libgfortran": "8.3.1",
       "libglvnd-egl": "1.0.1",
       "libglvnd-glx": "1.0.1",
       "libstdc++": "8.3.1",
       "libuuid": "2.32.1",
       "libwayland-client": "1.15.0",
       "libwayland-cursor": "1.15.0",
       "libwayland-egl": "1.15.0",
       "libX11": "1.6.7",
       "libX11-xcb": "1.6.7",
       "libxcb": "1.13",
       "libXcomposite": "0.4.4",
       "libxcrypt": "4.1.1",
       "libXcursor": "1.1.15",
       "libXdamage": "1.1.4",
       "libXext": "1.3.3",
       "libXfixes": "5.0.3",
       "libXi": "1.7.9",
       "libxkbcommon": "0.8.2",
       "libxkbcommon-x11": "0.7.1",
       "libXrandr": "1.5.1",
       "libXrender": "0.9.10",
       "libXtst": "1.2.3",
       "mesa-libGLU": "9.0.0",
       "mesa-libOSMesa": "19.1.4",
       "nspr": "4.21.0",
       "nss": "3.44.0",
       "nss-util": "3.44.0",
       "openjpeg2": "2.3.1",
       "openssl-libs": "1.1.1c",
       "pango": "1.42.4",
       "pulseaudio-libs": "11.1",
       "pulseaudio-libs-glib2": "11.1",
       "sqlite": "3.26.0",
       "xcb-util-image": "0.4.0",
       "xcb-util-keysyms": "0.4.0",
       "xcb-util-renderutil": "0.4.0",
       "xcb-util-wm": "0.4.0",
       "xdg-utils": "1.1.2",
       "xz-libs": "5.2.4",
       "zlib": "1.2.11",
    }
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
    if error or len(sys.argv) > 3 or os_version not in CENTOS_DEPENDENCIES:
        print(f'Usage: {sys.argv[0]} CentOS-version [build-type]', file=sys.stderr)
        print('  Supported CentOS versions are:', ', '.join(CENTOS_DEPENDENCIES.keys()),
              file=sys.stderr)
        print('  Build-type is one of "release", "candidate", or "daily"', file=sys.stderr)
        raise SystemExit(2)
    dependencies = CENTOS_DEPENDENCIES[os_version]
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
    bin_name = f"{app_name.lower()}"
    global APP_DIR
    if build == 'daily':
        # daily build, version is date
        version = version_date
        APP_DIR += "-daily"
        pkg_name += "-daily"
        bin_name += "-daily"
        rpm_release = 1
    elif build == 'techpreview':
        # like daily build, version is date
        version = version_date
        APP_DIR += "-techpreview"
        pkg_name += "-techpreview"
        bin_name += "-techpreview"
        rpm_release = 1
    elif build == 'release':
        # release build
        version = version.base_version
        rpm_release = 1
    elif build == 'candidate':
        # release build
        version = version.base_version
        rpm_release = f"0.{version_date}"
    bin_path = f"{PREFIX}/bin/{bin_name}"  # were the symlink is placed on default path

    # rpm_name = f"{pkg_name}-{version}"  # name of .rpm file
    # print('full_version:', repr(full_version))
    # print('version_number:', version_number)
    # print('version_date:', version_date)
    # print('rpm_name:', rpm_name)
    # print('pkg_name:', pkg_name)
    # print('bin_path:', bin_path)

    os.umask(0o22)  # turn off group and other writability
    rpmbuild_dir = os.path.expanduser("~/rpmbuild")
    make_rpmbuild_tree()
    clean_app()
    make_spec_file(rpmbuild_dir, pkg_name, version, rpm_release, bin_path, dependencies)

    subprocess.check_call([
        'rpmbuild',
        '-bb',
        '--nodeps',
        f"{rpmbuild_dir}/SPECS/{pkg_name}.spec",
    ])


def make_rpmbuild_tree():
    """Make packaging workspace directories
    """
    subprocess.check_call(["rpmdev-setuptree"])


def clean_app():
    """Clean application

    remove unwanted __pycache__ directories
    remove script's who interpreter is not a system binary
    (eg., Python scripts with paths to "nonexisting" python)
    """
    import shutil
    # cleanup -- remove __pycache__ directories
    cache_dirs = subprocess.check_output([
        '/usr/bin/find',
        CHIMERAX_INSTALL,
        '-name', '__pycache__'
        ]).strip().decode().split('\n')
    for d in cache_dirs:
        shutil.rmtree(d)

    # cleanup -- remove python shell scripts
    for fn in os.listdir(f'{CHIMERAX_INSTALL}/bin'):
        filename = f'{CHIMERAX_INSTALL}/bin/{fn}'
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


def make_spec_file(rpmbuild_dir, pkg_name, version, rpm_release, bin_path, dependencies):
    """Make control file"""
    if dependencies is not None:
        deps = list(dependencies)
    else:
        deps = []
    depends = ', '.join(deps)
    pkg_root = f'{PREFIX}/{APP_DIR}'
    bin_name = os.path.basename(bin_path)
    relpath = os.path.relpath(f'{pkg_root}/bin', os.path.dirname(bin_path))
    man_dir = f'{PREFIX}/share/man/man1'
    man_path = f'{man_dir}/{bin_name}.1'
    doc_dir = f'{PREFIX}/share/doc/{pkg_name}-{version}'
    with open(f"{rpmbuild_dir}/SPECS/{pkg_name}.spec", 'w') as f:
        #    %{{!?__debug_package:\
        #    /usr/lib/rpm/redhat/brp-strip %{{__strip}} \
        #    /usr/lib/rpm/redhat/brp-strip-comment-note %{{__strip}} %{{__objdump}} \
        #    }} \
        #    /usr/lib/rpm/redhat/brp-strip-static-archive %{{__strip}}
        print(textwrap.dedent(f"""\
            %define _build_id_links none
            %define __spec_install_post %{{nil}}
            %define debug_package %{{nil}}
            # %%define __os_install_post\
                /usr/lib/rpm/redhat/brp-compress

            AutoReqProv: no
            AutoReq: no
            AutoProv: no

            Name: {pkg_name}
            Version: {version}
            Release: {rpm_release}%{{?dist}}
            Summary: Molecular Visualization
            License: free for non-commercial use
            Vendor: Resource for Biocomputing, Visualization, and Informatics (RBVI)
            URL: https://www.rbvi.ucsf.edu/chimerax/
            ExclusiveArch: x86_64
            ExclusiveOS: linux
            Packager: Chimera Staff <chimera-staff@cgl.ucsf.edu>
            Group: Applications/Science
            # Suggests: ocl-icd
            Requires: {depends}
            Prefix: {PREFIX}

            %description
             UCSF ChimeraX (or simply ChimeraX) is the next-generation
             molecular visualization program from the Resource for Biocomputing
             Visualization, and Informatics (RBVI), following UCSF Chimera.
             ChimeraX can be downloaded free of charge for academic, government
             nonprofit, and personal use. Commercial users, please see licensing.

            %prep

            %build

            %install
            # copy application
            mkdir -p %{{buildroot}}{pkg_root}
            cp -a {CHIMERAX_INSTALL}/* %{{buildroot}}/{pkg_root}
            # make symbolic link to executable
            mkdir -p %{{buildroot}}{os.path.dirname(bin_path)}
            ln -s {relpath}/{app_name} %{{buildroot}}{bin_path}
            # make manual page
            mkdir -p %{{buildroot}}{man_dir}
            cp {CHIMERAX_INSTALL}/share/man/man1/{app_name}.1 %{{buildroot}}{man_path}
            gzip -n9 %{{buildroot}}{man_path}
            # make documentation
            mkdir -p %{{buildroot}}{doc_dir}
            cat > %{{buildroot}}{doc_dir}/NEWS << EOF
            "See https://www.rbvi.ucsf.edu/trac/ChimeraX/wiki/ChangeLog"
            EOF
            cat > %{{buildroot}}{doc_dir}/README << EOF
            Package is maintained by the ChimeraX team
            http://www.rbvi.ucsf.edu/chimerax

            Copyrights for embedded code are given in the documentation
            in {PREFIX}/{APP_DIR}/share/docs/embeded.html

            The computer code and documentation that comprises UCSF ChimeraX is protected
            by copyrights held by The Regents of the University of California ("The Regents")
            and is licensed for non-commercial use at no cost by The Regents.

            COMMERCIAL USE OF UCSF CHIMERAX IS COVERED BY A SEPARATE WRITTEN LICENSE AGREEMENT.
            Commercial licensing costs are tier-based, depending on the number of users.
            Please email chimerax@cgl.ucsf.edu
            if you are interested in using ChimeraX for commercial purposes.
            EOF

            %clean
            # rm -rf %{{buildroot}}

            %files
            %defattr(-,root,root,-)
            {pkg_root}
            {bin_path}
            {man_path}.gz
            {doc_dir}

            %post
            test -n "$RPM_INSTALL_PREFIX" || exit 1
            echo "Install desktop menu and associated mime types"
            $RPM_INSTALL_PREFIX/bin/{bin_name} --exit --nogui --silent --cmd 'linux xdg-install system true'
            echo "Precompiling Python packages"
            ($RPM_INSTALL_PREFIX/bin/{bin_name} -m compileall $RPM_INSTALL_PREFIX/{APP_DIR} || exit 0)

            %preun
            test -n "$RPM_INSTALL_PREFIX" || exit 1
            echo "Deregister desktop menu and associated mime types"
            $RPM_INSTALL_PREFIX/bin/{bin_name} --exit --nogui --silent --cmd 'linux xdg-uninstall system true'
            echo "Remove Python cache files"
            find $RPM_INSTALL_PREFIX/{APP_DIR} -name __pycache__ -print0 | xargs -0 /bin/rm -rf
            """), file=f)

        # Icon: .gif or .xpm!
        # Bugs: mailto:chimerax-bugs@cgl.ucsf.edu
        # Tags: science::visualisation, science::modelling, field::biology, field::chemistry,
        #  field::biology:structural, field::biology:bioinformatics,
        #  biology::nucleic-acids, biology::peptidic,
        #  scope::application, x11::application, role::program, interface::3d,
        #  implemented-in::python, uitoolkit::qt, use:viewing, network::client


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


if __name__ == '__main__':
    main()
