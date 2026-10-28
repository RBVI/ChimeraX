# Copyright (C) 2022 The Qt Company Ltd.
# SPDX-License-Identifier: LicenseRef-Qt-Commercial OR LGPL-3.0-only OR GPL-2.0-only OR GPL-3.0-only
# ChimeraX: PySide's signature_bootstrap.py with its module-level imports of
# traceback, zipfile, base64, io, pathlib and contextlib moved into the
# functions that use them.  ChimeraX ships shiboken6/files.dir, so the zip
# fallback is not used.
from __future__ import annotations

import os
import sys

from importlib.machinery import ModuleSpec


recursion_trap = 0


def bootstrap():

    global recursion_trap
    if recursion_trap:
        # we are probably called from outside, already
        print("Recursion occurred in Bootstrap. Did you start by hand? Then it's ok.")
        print("But you should trigger start by '_init_pyside_extension()', only!")
    recursion_trap += 1

    def import_loader(target, support_path):
        # Make sure that we always have the shibokensupport containing package first.
        # Also remove any prior loaded module of this name, just in case.
        target.insert(0, support_path)

        sbks = "shibokensupport"
        if sbks in sys.modules:
            del sys.modules[sbks]
        prefix = sbks + "."
        for key in list(key for key in sys.modules if key.startswith(prefix)):
            del sys.modules[key]
        try:
            import shibokensupport  # noqa: F401
            from shibokensupport.signature import loader
        except Exception as e:
            import traceback
            f = sys.stderr
            print("Problem importing shibokensupport:", file=f)
            print(f"{e.__class__.__name__}: {e}", file=f)
            traceback.print_exc()
            print("sys.path:", file=f)
            for p in sys.path:
                print("  " + p, file=f)
            f.flush()
            sys.exit(-1)
        target.remove(support_path)
        return loader

    incarnated = find_incarnated_files()
    if incarnated:
        target, support_path = sys.path, os.fspath(incarnated)
    else:
        target, support_path = prepare_zipfile()
    return import_loader(target, support_path)


def find_incarnated_files():
    import shiboken6 as root
    files_dir = os.path.join(os.path.dirname(os.path.realpath(root.__file__)), "files.dir")
    handle_embedding_switch(files_dir)
    if os.path.isdir(files_dir):
        sys.path.insert(0, files_dir)
        # Preload the loader to avoid recursion; see post_init_func in signature_globals.cpp.
        import shibokensupport.signature.loader  # noqa: F401
        del sys.path[0]
        return files_dir
    return None


def handle_embedding_switch(files_dir):
    env_var = os.environ.get("SBK_EMBED")
    if not env_var:
        return
    if env_var.lower() in ("1", "t", "true", "y", "yes"):
        import shutil
        shutil.rmtree(files_dir, ignore_errors=True)
    elif env_var.lower() in ("0", "f", "false", "n", "no"):
        reincarnate_files(files_dir)


def reincarnate_files(files_dir):
    from pathlib import Path
    files_dir = Path(files_dir)
    target, zip = prepare_zipfile()
    names = (_ for _ in zip.zfile.namelist() if _.endswith(".py"))
    try:
        files_dir.mkdir(exist_ok=True)
    except os.error:
        print(f"SBK_EMBED=False: Warning: Cannot write into {files_dir}")
        return None
    try:
        zip.zfile.extractall(path=files_dir, members=names)
        return files_dir
    except Exception as e:
        import traceback
        print(f"{e.__class__.__name__}: {e}", file=sys.stderr)
        traceback.print_exc()
        raise


def prepare_zipfile():
    import base64
    import io
    import zipfile
    # 'zipstring_sequence' comes from signature.cpp
    zipbytes = base64.b64decode(''.join(zipstring_sequence))   # noqa: F821
    vzip = zipfile.ZipFile(io.BytesIO(zipbytes))
    return sys.meta_path, EmbeddableZipImporter(vzip)


class EmbeddableZipImporter:

    def __init__(self, zip_file):
        def p2m(filename):
            if filename.endswith("/__init__.py"):
                return filename[:-12].replace("/", ".")
            if filename.endswith(".py"):
                return filename[:-3].replace("/", ".")
            return None

        self.zfile = zip_file
        self._mod2path = {p2m(_.filename): _.filename for _ in zip_file.filelist}

    def find_spec(self, fullname, path, target=None):
        path = self._mod2path.get(fullname)
        return ModuleSpec(fullname, self) if path else None

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        fullname = module.__spec__.name
        filename = self._mod2path[fullname]
        with self.zfile.open(filename, "r") as f:   # "rb" not for zipfile
            codeob = compile(f.read(), filename, "exec")
            exec(codeob, module.__dict__)
        module.__file__ = filename
        module.__loader__ = self
        if filename.endswith("/__init__.py"):
            module.__path__ = []
            module.__package__ = fullname
        else:
            module.__package__ = fullname.rpartition('.')[0]
        sys.modules[fullname] = module

# eof
