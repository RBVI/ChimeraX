# vi: set expandtab shiftwidth=4 softtabstop=4:

import getopt
import os
import shutil
import subprocess
import sys


verbose = False
my_name = None
skip = set()

DOCS = os.path.join("..", "docs")
BUNDLES = os.path.join("..", "src", "bundles")


def get_bundle_subdirs():
    # Use make to find out which bundles are included in build
    output = subprocess.check_output(["make", "-qp"], cwd=BUNDLES, encoding='utf-8')
    lines = output.split('\n')
    rest = None
    for line in lines:
        if line.startswith('REST_SUBDIRS ='):
            rest = line
            break
    return rest.split()[2:]


def build():
    _symlink_user(DOCS, True)
    for dirname in get_bundle_subdirs():
        # dirname may not be a directory, but we do not care
        # since test will return False anyway
        docs_dir = os.path.join(BUNDLES, dirname, "src", "docs")
        if os.path.exists(docs_dir):
            _symlink_user(docs_dir, False)
    generate_user_index(DOCS)


def generate_user_index(root):
    from chimerax.help_viewer.cmd import _generate_index
    index = os.path.join(root, 'user', 'index.html')
    generated = _generate_index(index, session.logger)
    os.remove('user/index.html')
    shutil.copyfile(generated, 'user/index.html')


def _symlink_user(root, conflict_fatal):

    user_path = os.path.join(root, "user")
    strip_length = len(user_path) - len("user")
    for dirpath, dirnames, filenames in os.walk(user_path):
        # Make sure directory in virtual docs exists
        my_dir = dirpath[strip_length:]
        if not os.path.exists(my_dir):
            _make_directory(my_dir)
        else:
            _check_directory(my_dir)

        # Get the right relative source directory path
        prefix = []
        head = my_dir
        while head:
            head, tail = os.path.split(head)
            if not head and not tail:
                break
            prefix.append("..")
        prefix.append(dirpath)
        src_dir = os.path.join(*prefix)

        # Create symlinks for each entry on remote end
        # Skip Makefiles since make must be run in true location, not here
        for filename in filenames:
            if filename in skip or filename[0] == '.':
                continue
            src = os.path.join(src_dir, filename)
            dst = os.path.join(my_dir, filename)
            if not os.path.exists(dst):
                _make_symlink(src, dst)
            else:
                _check_symlink(src, dst, conflict_fatal)


def _check_symlink(src, dst, conflict_fatal):
    if not os.path.islink(dst):
        print("%s: already exists and is not a symlink" % dst)
        raise SystemExit(1)
    osrc = os.readlink(dst)
    if src != osrc:
        print("%s: already exists and links to %s, not %s" % (dst, osrc, src))
        if conflict_fatal:
            raise SystemExit(1)


def _make_symlink(src, dst):
    try:
        os.symlink(src, dst)
    except OSError as e:
        print("%s: symlink %s: %s" % (src, dst, str(e)))
        raise SystemExit(1)


def _check_directory(p):
    if not os.path.isdir(p):
        print("%s: already exists and is not a directory" % p)
        raise SystemExit(1)


def _make_directory(p):
    try:
        os.mkdir(p)
    except OSError as e:
        print("%s: mkdir: %s" % (p, str(e)))
        raise SystemExit(1)


def check():
    bad_files = []
    for dirpath, dirnames, filenames in os.walk("."):
        for filename in filenames:
            if filename in skip or filename[0] == '.':
                continue
            filepath = os.path.join(dirpath, filename)
            if not os.path.islink(filepath):
                bad_files.append(os.path.relpath(filepath))
    try:
        bad_files.remove('user/index.html')
    except ValueError:
        pass
    if bad_files:
        if verbose:
            print("%d non-symlink files:" % len(bad_files))
            for filename in bad_files:
                print(" ", filename)
        raise SystemExit(1)


def clean():
    bad_files = []
    index_file = 'user/index.html'
    try:
        os.remove(index_file)
    except OSError as e:
        print("%s: %s" % (index_file, str(e)))
    for dirpath, dirnames, filenames in os.walk(".", topdown=False):
        for filename in filenames:
            if filename in skip or filename[0] == '.':
                continue
            filepath = os.path.join(dirpath, filename)
            if os.path.islink(filepath):
                _remove_symlink(filepath)
            else:
                bad_files.append(os.path.relpath(filepath))
        for dirname in dirnames:
            _remove_directory(os.path.join(dirpath, dirname))
    if bad_files:
        print("%d non-symlink files (not removed):" % len(bad_files))
        for filename in bad_files:
            print(" ", filename)
        raise SystemExit(1)


def _remove_symlink(p):
    try:
        os.remove(p)
    except OSError as e:
        print("%s: %s" % (p, str(e)))


def _remove_directory(p):
    try:
        os.rmdir(p)
    except OSError:
        pass


action = {
    "build": build,
    "check": check,
    "clean": clean,
}


def usage():
    print("Usage: %s (build|check|clean) [options...]" % my_name)
    raise SystemExit(2)


if __name__ == "__main__" or __name__.startswith("ChimeraX_sandbox"):
    my_name = os.path.basename(sys.argv[0])
    skip = set([my_name, "Makefile"])
    try:
        op = sys.argv[1]
    except IndexError:
        usage()
    opts, args = getopt.getopt(sys.argv[2:], "v")
    for opt, arg in opts:
        if opt == "-v":
            verbose = True
    try:
        action[op]()
    except KeyError as err:
        print('Error:', err)
        usage()
