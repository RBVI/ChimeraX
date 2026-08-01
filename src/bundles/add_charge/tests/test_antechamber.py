import os
import subprocess
import tempfile


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
        print("command: %s" % " ".join(str(a) for a in getattr(proc, "args", [])))
        print("exit code: %s" % rc)
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
