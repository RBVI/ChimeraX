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
import tempfile
import textwrap

# app_author and app_name are the same as ChimeraX_main.py
app_author = "UCSF"
app_name = "ChimeraX"

CHIMERAX_INSTALL = f"{os.getcwd()}/{app_name}.app"
CHIMERAX_BIN = f"{CHIMERAX_INSTALL}/bin/{app_name}"

INST_DIR = f"/opt/{app_author}/{app_name}"
#INST_DIR = "/usr/libexec/{app_author}-{app_name}""

CENTOS_DEPENDENCIES = {
    "7": {
        "alsa-lib": "1.1.4.1",
        "atk": "2.22.0",
        "cairo": "1.14.8",
        "cairo-gobject": "1.14.8",
        "cups-libs": "1.6.3",
        "dbus-libs": "1.10.24",
        "expat": "2.1.0",
        "fftw-libs-single": "3.3.3",
        "fontconfig": "2.10.95",
        "freetype": "2.4.11",
        "gdk-pixbuf2": "2.36.5",
        "glib2": "2.54.2",
        "glibc": "2.17",
        "gtk3": "3.22.26",
        "libdrm": "2.4.83",
        "libgcc": "4.8.5",
        "libstdc++": "4.8.5",
        "libX11": "1.6.5",
        "libxcb": "1.12",
        "libXcomposite": "0.4.4",
        "libXcursor": "1.1.14",
        "libXdamage": "1.1.4",
        "libXext": "1.3.3",
        "libXfixes": "5.0.3",
        "libXi": "1.7.9",
        "libXrandr": "1.5.1",
        "libXrender": "0.9.10",
        "libXtst": "1.2.3",
        "mesa-libEGL": "17.2.3",
        "mesa-libGL": "17.2.3",
        "mesa-libGLU": "9.0.0",
        "mesa-private-llvm": "3.9.1",
        # "mysql-community-libs": "5.6.40", or "mariadb-libs": "5.5.56", use: "mysql-libs"
        "mysql-libs": "",
        "nspr": "4.19.0",
        "nss": "3.36.0",
        "nss-util": "3.36.0",
        "openssl": "1.0.2k",
        "pango": "1.40.4",
        "pulseaudio-libs": "10.0",
        "xdg-utils": "1.1.0",
        "xz-libs": "5.2.2",
        "zlib": "1.2.7",
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
        if build not in ['release', 'candidate', 'daily']:
            error = True
    if error or len(sys.argv) > 3 or os_version not in CENTOS_DEPENDENCIES:
        print(f'Usage: {sys.argv[0]} CentOS-version [build-type]', file=sys.stderr)
        print('  Supported CentOS versions are:', ', '.join(CENTOS_DEPENDENCIES.keys()),
              file=sys.stderr)
        print('  Build-type is one of "release", "candidate", or "daily"', file=sys.stderr)
        raise SystemExit(2)
    dependencies = CENTOS_DEPENDENCIES[os_version]
    full_version = subprocess.check_output([
        CHIMERAX_BIN, "--nocolor", "--version"], stderr=subprocess.DEVNULL).decode()
    full_version = full_version.strip().split('\n')[1]
    full_version = full_version.split(':', maxsplit=1)[1].strip()
    version_number, version_date = full_version.split(maxsplit=1)
    version_date = version_date[1:-1].replace('-', '.')
    pkg_name = f"{app_author.lower()}-{app_name.lower()}"
    bin_path = f"/usr/bin/{app_name.lower()}"  # were the symlink is place on default path
    if build == 'daily':
        # daily build, version is date
        version = version_date
        global INST_DIR
        INST_DIR += "-daily"
        pkg_name += "-daily"
        bin_path += "-daily"
        rpm_release = 1
    elif build == 'release':
        # release build
        version = version_number
        rpm_release = 1
    elif build == 'candidate':
        # release build
        version = version_number
        rpm_release = f"0.{version_date}"
    rpm_name = f"{pkg_name}-{version}"  # name of .rpm file

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
    pkg_root = f'{INST_DIR}'
    bin_name = os.path.basename(bin_path)
    relpath = os.path.relpath(f'{pkg_root}/bin', os.path.dirname(bin_path))
    man_dir = '/usr/share/man/man1'
    man_path = f'{man_dir}/{bin_name}.1'
    doc_dir = f'/usr/share/doc/{pkg_name}-{version}'
    with open(f"{rpmbuild_dir}/SPECS/{pkg_name}.spec", 'w') as f:
        #    %{{!?__debug_package:\
        #    /usr/lib/rpm/redhat/brp-strip %{{__strip}} \
        #    /usr/lib/rpm/redhat/brp-strip-comment-note %{{__strip}} %{{__objdump}} \
        #    }} \
        #    /usr/lib/rpm/redhat/brp-strip-static-archive %{{__strip}} 
        print(textwrap.dedent(f"""\
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
            in {INST_DIR}/share/docs/embeded.html

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
            echo "Install desktop menu and associated mime types"
            {bin_path} --exit --nogui --silent --cmd 'linux xdg-install system true'
            echo "Precompiling Python packages"
            ({bin_path} -m compileall {pkg_root} || exit 0)

            %preun
            echo "Deregister desktop menu and associated mime types"
            {bin_path} --exit --nogui --silent --cmd 'linux xdg-uninstall system true'
            echo "Remove Python cache files"
            find {pkg_root} -name __pycache__ -print0 | xargs -0 /bin/rm -rf
            """), file=f)

        #Icon: .gif or .xpm!
        #Bugs: mailto:chimerax-bugs@cgl.ucsf.edu
        #Tags: science::visualisation, science::modelling, field::biology, field::chemistry,
        # field::biology:structural, field::biology:bioinformatics,
        # biology::nucleic-acids, biology::peptidic,
        # scope::application, x11::application, role::program, interface::3d,
        # implemented-in::python, uitoolkit::qt, use:viewing, network::client


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
