"""
ADIOS2 BP5 reader entry point.

Opens a BP5 archive and returns a BP5Trajectory model attached to the session.
If the archive contains topology information (atom names, residue names) it is
used to build a Structure; otherwise a stub structure is created.

The open_bp5 function follows the ChimeraX open-command provider protocol:
it returns (models, status_message).
"""

from __future__ import annotations

import os


def open_bp5(session, data, file_name: str, buffer_steps: int = 10,
             variable: str = "coordinates", topology_file: str = ""):
    """
    Open an ADIOS2 BP5 trajectory file.

    Parameters
    ----------
    session:
        Active ChimeraX session.
    data:
        File-like object or path provided by the open command manager.
    file_name:
        Path to the .bp directory/file.
    buffer_steps:
        Number of timesteps to keep in the ring buffer at once.
    variable:
        Name of the ADIOS2 variable holding coordinates (shape Natoms×3,
        typically float32 or float64 in the archive).
    topology_file:
        Optional path to a topology file (PDB/mmCIF) for atom names.
        If empty, a generic N-atom structure is generated.
    """
    _require_adios2()

    path = str(data) if hasattr(data, '__fspath__') else file_name

    from .trajectory import BP5Trajectory
    traj = BP5Trajectory(
        session,
        path,
        variable=variable,
        buffer_steps=buffer_steps,
        topology_file=topology_file,
    )

    n_atoms = traj.n_atoms
    n_steps = traj.n_steps
    status = (
        f"Opened BP5 trajectory '{os.path.basename(path)}': "
        f"{n_atoms} atoms, {n_steps} steps, "
        f"ring buffer {buffer_steps} steps."
    )
    return [traj], status


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _require_adios2():
    """Raise a clear UserError if ADIOS2 is not installed."""
    try:
        import adios2  # noqa: F401
    except ImportError:
        from chimerax.core.errors import UserError
        raise UserError(
            "ADIOS2 Python bindings are required to open BP5 trajectories.\n"
            "Install with:  pip install adios2\n"
            "           or: conda install -c conda-forge adios2"
        )
