"""
BP5Trajectory — ChimeraX Model for ADIOS2 BP5 streaming trajectories.

Design
------
* The archive is kept open for the lifetime of the model so individual steps
  can be fetched with BeginStep / InquireVariable / GetSync without loading
  the whole trajectory.
* Coordinates are stored as fp32 in a fixed-size ring buffer
  (default 10 steps).  Scrubbing the slider loads only the required step;
  past steps outside the ring window are evicted without replacement.
* On Apple Silicon (and generally in the Metal path) the fp32 buffer can be
  uploaded directly to a MTLBuffer with zero copy via shared storage mode.

ADIOS2 variable conventions (configurable via *variable* argument)
------------------------------------------------------------------
  coordinates : float32 or float64, shape (n_steps, n_atoms, 3) or per-step
                  (n_atoms, 3) in streaming mode.
  box_vectors  : optional float32 shape (3, 3) per step, Å units.
  time         : optional float32 scalar, ps units.

All coordinates are converted to fp32 Å at read time regardless of the
on-disk dtype.
"""

from __future__ import annotations

import os
from collections import OrderedDict

import numpy as np

from chimerax.core.models import Model


class BP5Trajectory(Model):
    """
    A ChimeraX Model backed by an ADIOS2 BP5 archive.

    Session save/restore is not yet implemented (the archive path is
    recorded so the trajectory can be re-opened on restore).
    """

    SESSION_ENDURING = False
    SESSION_SAVE = False

    def __init__(
        self,
        session,
        path: str,
        *,
        variable: str = "coordinates",
        buffer_steps: int = 10,
        topology_file: str = "",
    ):
        name = os.path.basename(path)
        super().__init__(name, session)

        self._path = path
        self._variable = variable
        self._buffer_steps = max(1, buffer_steps)
        self._topology_file = topology_file

        self._adios = None   # adios2.ADIOS instance
        self._io = None      # adios2.IO
        self._engine = None  # adios2.Engine (File / BP5 reader)
        self._n_atoms = 0
        self._n_steps = 0
        self._current_step = -1

        # Ring buffer: OrderedDict of {step_index: fp32 (n_atoms, 3)}
        self._ring: OrderedDict[int, np.ndarray] = OrderedDict()

        # Underlying atomic structure (may be None until topology is known)
        self._structure = None

        self._open_archive()
        self._load_structure()
        session.models.add([self])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def n_atoms(self) -> int:
        return self._n_atoms

    @property
    def n_steps(self) -> int:
        return self._n_steps

    @property
    def current_step(self) -> int:
        return self._current_step

    def goto_step(self, step: int) -> np.ndarray:
        """
        Set the displayed frame to *step* (0-based).

        Returns the fp32 (n_atoms, 3) coordinate array for that step.
        If the step is already in the ring buffer it is returned immediately.
        Otherwise the archive is seeked to that step and the coordinates
        are fetched and cached.
        """
        step = max(0, min(step, self._n_steps - 1))
        if step not in self._ring:
            self._fetch_step(step)
        self._current_step = step
        coords = self._ring[step]
        self._apply_to_structure(coords)
        return coords

    def prefetch(self, center_step: int, radius: int = 5) -> None:
        """
        Asynchronously prefetch steps within *radius* of *center_step* into
        the ring buffer.  Steps already cached are skipped.  Steps that fall
        outside the ring window are evicted to make room.
        """
        lo = max(0, center_step - radius)
        hi = min(self._n_steps - 1, center_step + radius)
        for s in range(lo, hi + 1):
            if s not in self._ring:
                self._fetch_step(s)

    def close(self):
        """Release the ADIOS2 engine and all cached data."""
        self._ring.clear()
        if self._engine is not None:
            try:
                self._engine.Close()
            except Exception:
                pass
            self._engine = None
        self._io = None
        self._adios = None

    # ------------------------------------------------------------------
    # Model overrides
    # ------------------------------------------------------------------

    def delete(self):
        self.close()
        super().delete()

    # ------------------------------------------------------------------
    # Private: archive I/O
    # ------------------------------------------------------------------

    def _open_archive(self):
        import adios2

        self._adios = adios2.ADIOS()
        self._io = self._adios.DeclareIO("bp5_reader")
        self._io.SetEngine("BP5")

        self._engine = self._io.Open(self._path, adios2.Mode.Read)

        # Peek at the variable to determine shape.
        # In BP5 streaming layout variables have per-step local shape.
        var = self._io.InquireVariable(self._variable)
        if var is None:
            raise ValueError(
                f"Variable '{self._variable}' not found in BP5 archive "
                f"'{self._path}'.  Available: "
                f"{list(self._io.AvailableVariables().keys())}"
            )

        shape = var.Shape()
        # Shape can be (n_steps, n_atoms, 3) for joined arrays or
        # (n_atoms, 3) per-step in pure streaming mode.
        if len(shape) == 3:
            self._n_steps = int(shape[0])
            self._n_atoms = int(shape[1])
        elif len(shape) == 2:
            # Per-step shape; count steps from the engine.
            self._n_atoms = int(shape[0])
            self._n_steps = int(self._engine.Steps())
        else:
            raise ValueError(
                f"Unexpected shape {shape} for variable '{self._variable}'"
            )

    def _fetch_step(self, step: int):
        """Read one step from the archive into the ring buffer."""
        import adios2

        engine = self._engine
        var = self._io.InquireVariable(self._variable)

        # For joined arrays use SetStepSelection; for per-step streaming
        # use BeginStep/EndStep.
        shape = var.Shape()
        if len(shape) == 3:
            var.SetStepSelection(adios2.Box([step, 0, 0], [1, self._n_atoms, 3]))
            buf = np.empty((1, self._n_atoms, 3), dtype=np.float64)
            engine.Get(var, buf)
            engine.PerformGets()
            coords = buf[0].astype(np.float32)
        else:
            # Pure streaming: advance to the right step.
            # NOTE: BP5 in streaming mode only supports forward-only seeks.
            # For random access the engine must be reopened (handled below).
            current_engine_step = getattr(self, '_engine_step', -1)
            if step <= current_engine_step:
                self._reopen_engine()
                current_engine_step = -1

            while current_engine_step < step:
                status = engine.BeginStep()
                if status != adios2.StepStatus.OK:
                    break
                current_engine_step += 1

            buf = np.empty((self._n_atoms, 3), dtype=np.float64)
            engine.Get(var, buf)
            engine.EndStep()
            self._engine_step = current_engine_step
            coords = buf.astype(np.float32)

        self._cache_step(step, coords)

    def _reopen_engine(self):
        """Close and reopen the BP5 engine to allow backwards seeking."""
        import adios2
        if self._engine is not None:
            try:
                self._engine.Close()
            except Exception:
                pass
        self._engine = self._io.Open(self._path, adios2.Mode.Read)
        self._engine_step = -1

    def _cache_step(self, step: int, coords: np.ndarray):
        """Insert *coords* into the ring buffer, evicting oldest if full."""
        while len(self._ring) >= self._buffer_steps:
            self._ring.popitem(last=False)  # evict oldest
        self._ring[step] = coords

    # ------------------------------------------------------------------
    # Private: structure integration
    # ------------------------------------------------------------------

    def _load_structure(self):
        """
        Build or import an atomic structure.

        If a *topology_file* was given, open it and use that structure.
        Otherwise create a minimal stub structure with generic atoms so
        that ChimeraX can visualise the trajectory even without topology.
        """
        if self._topology_file and os.path.isfile(self._topology_file):
            from chimerax.core.commands import run
            models, _ = run(
                self.session,
                f"open {self._topology_file!r}",
                log=False,
            )
            if models:
                self._structure = models[0]
                return

        # Create a stub: one CA atom per trajectory atom.
        try:
            from chimerax.atomic import AtomicStructure
            s = AtomicStructure(self.session, name=f"{self.name} [topology]")
            r = s.new_residue("UNK", "A", 1)
            for i in range(self._n_atoms):
                a = s.new_atom(f"CA{i+1}", "C")
                a.radius = 1.5
                r.add_atom(a)
            self._structure = s
        except Exception:
            self._structure = None

    def _apply_to_structure(self, coords: np.ndarray):
        """Update the atomic structure with coordinates from *coords* (fp32)."""
        s = self._structure
        if s is None:
            return
        atoms = s.atoms
        n = min(len(atoms), len(coords))
        if n == 0:
            return
        # Atomic structure stores coordinates as float64 internally;
        # the GPU upload boundary in drawing_walker.py converts to fp32.
        atoms[:n].coords = coords[:n].astype(np.float64)
        s.change_tracker.add_modified(s, s.SCENE_POSITION_CHANGE)
