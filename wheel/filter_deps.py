# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2023 Regents of the University of California.
# All rights reserved.  This software provided pursuant to a
# license agreement containing restrictions on its disclosure,
# duplication and use.  For details see:
# http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html
# This notice must be embedded in or attached to all copies,
# including partial copies, of the software or any revisions
# or derivations thereof.
# === UCSF ChimeraX Copyright ===
import os

from packaging.requirements import Requirement

dependencies = []
ignored_gui_dependencies = {
    "qtconsole"
    , "openvr"
}
ignored_prefixes = ["app_pyopengl", "build", "dev"]
dependency_files = [
    file for file in os.listdir("../prereqs/pips")
    if ('requirements' in file) and not any([prefix in file for prefix in ignored_prefixes])
]
for file in dependency_files:
    with open("../prereqs/pips/" + file) as f:
        for line in f:
            if not line.startswith("#") and line != "\n":
                tmp = Requirement(line.strip())
                if tmp.name not in ignored_gui_dependencies:
                    dependencies.append('"' + line.strip() + '"')

with open("pyproject.toml.in") as f:
    content = f.read()
with open("pyproject.toml", "w") as f:
    f.write(content.replace("NO_GUI_DEPENDENCIES", ",\n\t".join(dependencies)))
raise SystemExit(0)
