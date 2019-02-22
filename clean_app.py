#
# vi:set expandtab shiftwidth=4:
#
# Post-install cleanup up of application directory
#
import sys
import os


def clean_app(chimerax_root):
    """Clean application
    
    remove unwanted __pycache__ directories
    remove script's whose interpreter is not a system binary
    (eg., Python scripts with paths to "nonexisting" python)
    """
    import shutil
    # cleanup -- remove __pycache__ directories
    cache_dirs = []
    for root, dirs, files in os.walk(chimerax_root):
        if "__pycache__" in dirs:
            cache_dirs.append(os.path.join(root, "__pycache__"))
    for d in cache_dirs:
        shutil.rmtree(d)

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
