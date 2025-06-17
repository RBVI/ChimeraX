import os

packages = os.listdir('chimerax')

with open("chimerax/__init__.py", "w") as f:
    imports = [f'from . import {package} #noqa' for package in packages if "__pycache__" not in package]
    f.write("\n".join(imports))
