from sys import stderr
import os
from os import remove, chmod, rename
from os.path import join, exists, isdir, islink, isfile, getsize
from os import mkdir, listdir
from shutil import copy, copytree, copyfile
import subprocess
import lief
import sys


# -------------------------------------------------------------------------------------
# Make a Mac universal build by combining Arm and Intel builds using the lipo command.
#
def make_universal(
    arm_location, intel_location, universal_location, exclude, warn_on_mismatch
):
    paths = tree_files(arm_location, intel_location, exclude=exclude)

    if not exists(universal_location):

        mkdir(universal_location)
    for path in paths:
        apath, ipath, upath = (
            join(arm_location, path),
            join(intel_location, path),
            join(universal_location, path),
        )
        apath_exists, ipath_exists = exists(apath), exists(ipath)
        warn = warn_on_mismatch(path)
        if apath_exists and not ipath_exists:
            if warn:
                log_mismatch(f"Missing Intel {path}")
            _copy(apath, upath, whole_tree=True)
        elif ipath_exists and not apath_exists:
            if warn:
                log_mismatch(f"Missing ARM {path}")
            _copy(ipath, upath, whole_tree=True)
        elif same_file(apath, ipath):
            _copy(apath, upath)
        elif is_executable(apath):
            lipo_files(apath, ipath, upath, warn)
        else:
            if warn:
                log_mismatch(f"Differ {path}")
            # Use ARM if files differ.
            _copy(apath, upath)

    use_intel_info_plist(intel_location, universal_location)


def tree_files(path1, path2, exclude, prefix="", paths=None):

    files1, files2 = listdir(path1), listdir(path2)
    if paths is None:
        paths = []

    for file in set(files1 + files2):
        path = join(prefix, file)
        if exclude(path):
            continue
        paths.append(path)
        dir1, dir2 = join(path1, file), join(path2, file)
        if isdir(dir1) and not islink(dir1) and isdir(dir2) and not islink(dir2):
            tree_files(dir1, dir2, exclude, prefix=path, paths=paths)
    return paths


def _copy(file1, file2, whole_tree=False):

    if islink(file1) or isfile(file1):
        if not islink(file2) and not exists(file2):

            copy(file1, file2, follow_symlinks=False)
    elif isdir(file1) and not exists(file2):
        if whole_tree:

            copytree(file1, file2, symlinks=True)
        else:

            mkdir(file2)


def same_file(file1, file2):

    if islink(file1) and islink(file2):
        return True
    if isdir(file1) and isdir(file2):
        return True
    if isfile(file1) and isfile(file2):
        if not files_differ(file1, file2):
            return True
        if only_line_endings_differ(file1, file2):
            return True
    return False


def files_differ(arm_path, intel_path):

    if getsize(arm_path) != getsize(intel_path):
        return True
    with open(arm_path, "rb") as afile:
        with open(intel_path, "rb") as ifile:
            if afile.read() != ifile.read():
                return True
    return False


def only_line_endings_differ(arm_path, intel_path):
    with open(arm_path, "rb") as afile:
        with open(intel_path, "rb") as ifile:
            while True:
                aline, iline = afile.readline(), ifile.readline()
                if aline.rstrip() != iline.rstrip():
                    return False
                if len(aline) == 0 and len(iline) == 0:
                    break
    return True


need_lipo = set(
    [
        lief.MachO.Header.FILE_TYPE.BUNDLE,
        lief.MachO.Header.FILE_TYPE.DYLIB,
        lief.MachO.Header.FILE_TYPE.EXECUTE,
        lief.MachO.Header.FILE_TYPE.OBJECT,
    ]
)
lief.logging.disable()


def is_executable(path):

    if not lief.is_macho(path):
        return False
    try:
        m = lief.MachO.parse(path)
    # Bare except is usually bad, but lief does not provide its own
    # exceptions
    except Exception as e:
        return False
    if m is None or m.at(0) is None:
        return False
    file_type = m.at(0).header.file_type
    return file_type in need_lipo


def lipo_files(arm_path, intel_path, universal_path, warn):
    if not is_executable(intel_path) and warn:
        log_mismatch(f"Not executable {intel_path}")
        _copy(arm_path, universal_path)
        return

    # Use lipo command to merge ARM and Intel binaries.
    # The lipo command uses different options for handling thin versus fat files
    # so first we extract thin versions.
    try:
        arm_path_thin = universal_path + ".arm64_thin"
        if not make_thin(arm_path, arm_path_thin, "arm64"):
            log_mismatch(f"ARM ChimeraX has only Intel binary: {arm_path}")
            _copy(arm_path, universal_path)
            chmod(universal_path, 0o755)  # Add execute permission.
            return

        intel_path_thin = universal_path + ".x86_64_thin"
        if not make_thin(intel_path, intel_path_thin, "x86_64"):
            log_mismatch(f"Intel ChimeraX has non-Intel binary: {intel_path}")
            _copy(arm_path, universal_path)
            chmod(universal_path, 0o755)  # Add execute permission.
            return

        try:
            args = [
                "lipo",
                arm_path_thin,
                intel_path_thin,
                "-create",
                "-output",
                universal_path,
            ]
            p = subprocess.run(args, capture_output=True)
            if p.returncode != 0:
                cmd = " ".join(args)
                raise RuntimeError(
                    "Error in lipo command: %s\nstdout:\n%s\nstderr:\n%s"
                    % (cmd, p.stdout, p.stderr)
                )

            """
            from os.path import getsize, basename, dirname
            log_mismatch('lipo %d %d %d %s %s' %
                         (getsize(universal_path), getsize(arm_path_thin), getsize(intel_path_thin),
                          basename(universal_path), dirname(universal_path)))
            """

            remove(arm_path_thin)
            remove(intel_path_thin)
            chmod(universal_path, 0o755)  # Add execute permission.
        except RuntimeError as e:
            error = str(e)
            if "have the same architecture" in error:
                remove(arm_path_thin)
                rename(intel_path_thin, universal_path)
                stderr.write(
                    "both builds had same arch for %s; renaming instead of lipoing"
                    % universal_path
                )
            else:
                raise e
    except RuntimeError as e:
        error = str(e)
        if "does not contain the specified architecture (arm64)" in error:
            # if it's x86 then use it I guess
            if os.path.exists(intel_path):
                rename(intel_path, universal_path)
            elif os.path.exists(arm_path):
                rename(arm_path, universal_path)
            else:
                raise RuntimeError("No binary file %s" % universal_path)


def make_thin(path, thin_path, arch):
    args = ["lipo", path, "-info"]

    p = subprocess.run(args, capture_output=True)
    if p.returncode != 0:
        cmd = " ".join(args)
        raise RuntimeError(
            "Error in lipo command: %s\nstdout:\n%s\nstderr:\n%s"
            % (cmd, p.stdout, p.stderr)
        )
    if arch.encode("utf-8") not in p.stdout:
        return False  # binary does not contain desired architecture
    elif p.stdout.startswith(b"Non-fat"):

        copyfile(path, thin_path)
    else:
        args = ["lipo", path, "-thin", arch, "-output", thin_path]

    p = subprocess.run(args, capture_output=True)
    if p.returncode != 0:
        cmd = " ".join(args)
        raise RuntimeError(
            "Error in lipo command: %s\nstdout:\n%s\nstderr:\n%s"
            % (cmd, p.stdout, p.stderr)
        )
    return True


def use_intel_info_plist(intel_location, universal_location):
    """
    The ARM Info.plist specifies the minimum os version as 11.0 while
    the Intel specifies it as 10.13.  Use the older Intel version otherwise
    the app icon appears crossed-out and unrunnable on macOS 10.15 and older.
    """

    copyfile(
        join(intel_location, "Contents", "Info.plist"),
        join(universal_location, "Contents", "Info.plist"),
    )


def log_mismatch(message):

    stderr.write(message + "\n")


def has_suffix(path, suffixes):
    for suffix in suffixes:
        if path.endswith(suffix):
            return True
    return False


omit = [
    "_CodeSignature",
    "debugpy",
    ".a",
    "libtcl8.6.dylib",  # Causes notarization failure. Not used.
    "libtk8.6.dylib",  # Causes notarization failure. Not used.
    #    "libHoloPlayCore.dylib",  # 0.1.0 has no ARM symbols (looking_glass)
    #    "libopenvr_api_32.dylib",  # VR is not even supported on macOS
    #    # These amber libs are always amd64 even on arm64 macos
    #    "libgfortran.3.dylib",
    #    "libgcc_s.1.dylib",
    #    "libquadmath.0.dylib",
    #    # These amber binaries are always amd64 even on arm64 macos
    #    "sqm",
    #    "espgen",
    #    "am1bcc",
    #    "antechamber",
    #    "atomtype",
    #    "bondtype",
    #    "nc-config",
    #    "nf-config",
    #    "parmchk2",
    #    "prepgen",
    #    "residuegen",
    #    "respgen",
    #    ##### end of amber libs and binaries
    #    "python3-intel64",  # Comes with ARM Python but is not universal
    #    "python3.11-intel64",  # Comes with ARM Python but is not universal
]

no_warn = [
    "__pycache__",
    ".pyc",
    ".dist-info/RECORD",
    ".dist-info/WHEEL",
    ".dist-info/METADATA",
    ".dist-info/direct_url.json",
    "Contents/share/install-timestamp",
    "LICENSE.txt",
]


def exclude(path, omit=omit):
    return has_suffix(path, omit)


def warn_on_mismatch(path, no_warn=no_warn):
    return not has_suffix(path, no_warn)


arm_location, intel_location, universal_location = sys.argv[1:4]
make_universal(
    arm_location, intel_location, universal_location, exclude, warn_on_mismatch
)
