#
# vi:set expandtab shiftwidth=4:
#
# Post-install cleanup up of application directory
#
import sys
import os
import platform
import shutil
import subprocess


def clean_app(chimerax_root):
    """Clean application tree for distribution"""
    remove_pycache(chimerax_root)
    if sys.platform == "darwin":
        thin_macos_binaries(chimerax_root, platform.machine())
    # TODO: figure this out
    #   removing scripts makes multiple "make installs" without
    #   a "make clean" inbetween fail and removes scripts that
    #   would be needed for developers to make documentation
    # remove_scripts(chimerax_root)


def remove_pycache(chimerax_root):
    """remove unwanted __pycache__ directories"""
    # cleanup -- remove __pycache__ directories
    cache_dirs = []
    for root, dirs, files in os.walk(chimerax_root):
        if "__pycache__" in dirs:
            cache_dirs.append(os.path.join(root, "__pycache__"))
    for d in cache_dirs:
        shutil.rmtree(d)


def is_fat_macho(path):
    """Check for a universal binary's magic number (also used by Java class
    files, so confirm with lipo)"""
    with open(path, "rb") as f:
        return f.read(4) in (b"\xca\xfe\xba\xbe", b"\xca\xfe\xba\xbf")


def thin_macos_binaries(chimerax_root, arch):
    """Remove other architectures from universal binaries

    The Intel and Arm builds are merged into a universal application later,
    so each build only needs its own architecture.  Wheels like PySide6's
    are universal2, which roughly doubles their size.
    """
    freed = 0
    for root, dirs, files in os.walk(chimerax_root):
        for fn in files:
            path = os.path.join(root, fn)
            if os.path.islink(path) or not is_fat_macho(path):
                continue
            result = subprocess.run(["lipo", "-archs", path],
                                    capture_output=True, text=True)
            if result.returncode != 0:
                continue
            if arch not in result.stdout.split():
                print(f"clean_app: no {arch} code in {path}", file=sys.__stderr__)
                continue
            size = os.path.getsize(path)
            thin = path + ".thin"
            subprocess.run(["lipo", "-thin", arch, path, "-output", thin],
                           check=True)
            shutil.copymode(path, thin)
            os.replace(thin, path)
            freed += size - os.path.getsize(path)
    print(f"clean_app: removed {freed / 1e6:.0f} MB of non-{arch} code")


def remove_scripts(chimerax_root):
    """remove script's whose interpreter is not a system binary
    (eg., Python scripts with paths to "nonexisting" python)
    """
    # cleanup -- remove python shell scripts
    if sys.platform.startswith('win'):
        shutil.rmtree(f'{chimerax_root}/bin/Scripts')
    else:
        for fn in os.listdir(f'{chimerax_root}/bin'):
            filename = f'{chimerax_root}/bin/{fn}'
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


if __name__ == "__main__":
    bin_dir = os.path.dirname(sys.executable)
    if os.path.basename(bin_dir) != 'bin':
        print("Must be called as CHIMERAX/bin/python", file=sys.__stderr__)
        raise SystemExit(1)
    clean_app(os.path.dirname(bin_dir))
