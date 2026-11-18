#!/usr/bin/env python3
# vim: set expandtab shiftwidth=4 softtabstop=4:
"""Make a startup prefetch list to ship with ChimeraX.

The launcher reads ahead the files a start needs from a list that
chimerax.core.startup_prefetch records in the user's cache after each GUI
start.  Until the first start records one, the launchers use
a list shipped in the application's share directory instead, whose file
names are relative to the application directory: ChimeraX.app on Linux,
ChimeraX.app/Contents on macOS, or the launcher's directory on Windows.

Record a list by starting the built application once with its GUI for more
than 3 seconds, then convert it:

    make_startup_prefetch_list.py RECORDED_LIST APP_DIR APP_DIR/share/startup-prefetch

Entries outside APP_DIR, such as the user's settings and caches, are
dropped.  Byte ranges and other fields after a tab are kept.
"""
import os
import sys


def relative_list(text, app_dir):
    root = os.path.realpath(app_dir)
    lines = []
    for line in text.split("\n"):
        path, tab, rest = line.partition("\t")
        if not os.path.isabs(path) or "\r" in path:
            continue
        absolute = os.path.abspath(path)
        resolved = os.path.realpath(absolute)
        try:
            if os.path.normcase(os.path.commonpath((root, resolved))) != os.path.normcase(root):
                continue
            relative = os.path.relpath(absolute, app_dir)
        except ValueError:  # Different drives on Windows.
            continue
        if relative != os.curdir and not relative.startswith(os.pardir + os.sep):
            lines.append(relative + tab + rest)
    return "".join(line + "\n" for line in lines)


def main(argv):
    if len(argv) != 4:
        sys.exit(__doc__)
    recorded, app_dir, output = argv[1:]
    with open(recorded, encoding="utf-8") as f:
        text = relative_list(f.read(), app_dir)
    with open(output, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    print(f"{output}: {text.count(chr(10))} entries")


if __name__ == "__main__":
    main(sys.argv)
