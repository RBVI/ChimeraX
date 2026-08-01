import os
import struct
import subprocess
import sys
import tempfile


def _pe_imported_dlls(path):
    """Return the list of DLL names in a PE file's import table.

    A tiny hand-rolled PE parser so the dependency walk needs no external tools
    (dumpbin/ntldd may not be present on the CI runner).  Returns [] for anything
    that does not look like a parseable PE image.
    """
    with open(path, "rb") as f:
        data = f.read()
    if data[:2] != b"MZ":
        return []
    try:
        e_lfanew = struct.unpack_from("<I", data, 0x3C)[0]
        if data[e_lfanew:e_lfanew + 4] != b"PE\0\0":
            return []
        coff = e_lfanew + 4
        num_sections = struct.unpack_from("<H", data, coff + 2)[0]
        size_opt = struct.unpack_from("<H", data, coff + 16)[0]
        opt = coff + 20
        magic = struct.unpack_from("<H", data, opt)[0]
        if magic == 0x10B:        # PE32
            data_dirs = opt + 96
        elif magic == 0x20B:      # PE32+
            data_dirs = opt + 112
        else:
            return []
        # Data directory index 1 is the import table.
        import_rva = struct.unpack_from("<I", data, data_dirs + 8)[0]
        if import_rva == 0:
            return []
        sections = []
        sec_off = opt + size_opt
        for i in range(num_sections):
            base = sec_off + i * 40
            vsize = struct.unpack_from("<I", data, base + 8)[0]
            va = struct.unpack_from("<I", data, base + 12)[0]
            raw_size = struct.unpack_from("<I", data, base + 16)[0]
            raw = struct.unpack_from("<I", data, base + 20)[0]
            sections.append((va, max(vsize, raw_size), raw))

        def rva_to_off(rva):
            for va, span, raw in sections:
                if va <= rva < va + span:
                    return raw + (rva - va)
            return None

        names = []
        off = rva_to_off(import_rva)
        if off is None:
            return []
        # Walk the IMAGE_IMPORT_DESCRIPTOR array (20 bytes each) until the
        # all-zero terminator; the DLL name RVA is at offset 12.
        while True:
            descriptor = data[off:off + 20]
            if len(descriptor) < 20 or descriptor == b"\0" * 20:
                break
            name_rva = struct.unpack_from("<I", descriptor, 12)[0]
            if name_rva == 0:
                break
            noff = rva_to_off(name_rva)
            if noff is not None:
                end = data.find(b"\0", noff)
                if end != -1:
                    names.append(data[noff:end].decode("ascii", "replace"))
            off += 20
        return names
    except (struct.error, IndexError):
        return []


def _dll_search_dirs(exe):
    """Approximate the search order Windows used for antechamber's DLLs: the
    executable's own directory, the system directories, then PATH."""
    windir = os.environ.get("SystemRoot", r"C:\Windows")
    dirs = [os.path.dirname(exe),
        os.path.join(windir, "System32"),
        os.path.join(windir, "SysWOW64"),
        windir]
    dirs += os.environ.get("PATH", "").split(os.pathsep)
    return dirs


def _resolve_dll(name, dirs):
    for d in dirs:
        if not d:
            continue
        candidate = os.path.join(d, name)
        if os.path.isfile(candidate):
            return candidate
    return None


def _walk_dependencies(exe, max_files=300):
    """Recursively resolve exe's DLL imports.  Returns (resolved, missing) where
    resolved maps dll-name -> path|None and missing is a list of
    (dll_name, imported_by)."""
    dirs = _dll_search_dirs(exe)
    resolved = {}
    missing = []
    stack = [(os.path.basename(exe), exe)]
    while stack and len(resolved) < max_files:
        importer_name, importer_path = stack.pop()
        for dll in _pe_imported_dlls(importer_path):
            key = dll.lower()
            if key in resolved:
                continue
            path = _resolve_dll(dll, dirs)
            resolved[key] = path
            if path is None:
                missing.append((dll, importer_name))
            else:
                stack.append((dll, path))
    return resolved, missing


def _dump_dependencies(exe):
    """Print antechamber's DLL dependency tree and, crucially, any DLL that
    cannot be found on the search path -- the cause of a 0xC0000135 launch."""
    if not os.path.isfile(exe):
        print("cannot inspect dependencies: %s does not exist" % exe)
        return
    resolved, missing = _walk_dependencies(exe)
    print("dependency scan of %s:" % exe)
    for name in sorted(resolved):
        print("  %-32s %s" % (name, resolved[name] or "*** NOT FOUND ***"))
    if missing:
        print("MISSING DLLs (this is why antechamber could not launch):")
        for dll, importer in missing:
            note = ""
            if dll.lower().startswith(("api-ms-win-", "ext-ms-win-")):
                note = "  (API set; normally resolved virtually by the loader)"
            print("  %s  (imported by %s)%s" % (dll, importer, note))
    else:
        print("all imported DLLs resolved on the current search path")


def _dump_diagnostics(kept_dirs, procs):
    """Print everything useful for diagnosing an ANTECHAMBER failure.

    ANTECHAMBER runs as a subprocess whose stdout ChimeraX only routes to its
    own log, and it does its work in a TemporaryDirectory that is deleted before
    the test can look at it.  On CI (where the charge calculation fails but local
    and released builds succeed) that leaves nothing to go on.  This reconstructs
    the exit code and the working-directory contents so the failure is visible in
    the pytest output instead of an opaque "Check reply log for details".
    """
    print("\n===== ANTECHAMBER failure diagnostics =====")
    for proc in procs:
        # The process has already run to EOF by the time we get here, so a short
        # wait reaps it and yields the real exit code without risking a hang.  A
        # launch failure can leave it half-built, so guard the call.
        try:
            rc = proc.wait(timeout=10)
        except Exception as e:
            rc = "unavailable (%s)" % e
        args = getattr(proc, "args", [])
        print("command: %s" % " ".join(str(a) for a in args))
        print("exit code: %s" % rc)
        # 0xC0000135 (== 3221225781) is STATUS_DLL_NOT_FOUND: antechamber could
        # not load a dependency.  On Windows, walk its imports to name it.
        if sys.platform == "win32" and args:
            exe = str(args[0]).replace("/", os.sep)
            for candidate in (exe, exe + ".exe"):
                if os.path.isfile(candidate):
                    exe = candidate
                    break
            _dump_dependencies(exe)
    for d in kept_dirs:
        if not os.path.isdir(d):
            print("working directory %s no longer exists" % d)
            continue
        print("working directory: %s" % d)
        for name in sorted(os.listdir(d)):
            path = os.path.join(d, name)
            try:
                size = os.path.getsize(path)
            except OSError as e:
                print("  %s (could not stat: %s)" % (name, e))
                continue
            print("  --- %s (%d bytes) ---" % (name, size))
            try:
                with open(path, encoding="utf8", errors="replace") as f:
                    contents = f.read(20000)
            except (OSError, ValueError) as e:
                print("    (could not read: %s)" % e)
                continue
            for line in contents.splitlines():
                print("    %s" % line)
    print("===== end diagnostics =====\n")


def test_match_maker(test_production_session, monkeypatch):
    from chimerax.core.commands import run

    session = test_production_session

    kept_dirs = []
    antechamber_procs = []

    class _KeptTemporaryDirectory:
        """Stand-in for tempfile.TemporaryDirectory that never cleans up, so the
        ANTECHAMBER working files remain for inspection if the run fails."""

        def __init__(self, *args, **kwargs):
            # mkdtemp only accepts suffix/prefix/dir; drop TemporaryDirectory-only
            # kwargs like delete=/ignore_cleanup_errors= that other callers may pass.
            mkdtemp_kwargs = {k: v for k, v in kwargs.items()
                if k in ("suffix", "prefix", "dir")}
            self.name = tempfile.mkdtemp(*args, **mkdtemp_kwargs)
            kept_dirs.append(self.name)

        def __enter__(self):
            return self.name

        def __exit__(self, *exc_info):
            return False

        def cleanup(self):
            pass

    class _RecordingPopen(subprocess.Popen):
        def __init__(self, args, *rest, **kwargs):
            if any("antechamber" in str(a) for a in args):
                antechamber_procs.append(self)
            super().__init__(args, *rest, **kwargs)

    monkeypatch.setattr(tempfile, "TemporaryDirectory", _KeptTemporaryDirectory)
    monkeypatch.setattr(subprocess, "Popen", _RecordingPopen)

    run(session, "open 2gbp")
    run(session, "addh")
    try:
        run(session, "addcharge")
    except Exception:
        _dump_diagnostics(kept_dirs, antechamber_procs)
        raise
    finally:
        for d in kept_dirs:
            import shutil

            shutil.rmtree(d, ignore_errors=True)
