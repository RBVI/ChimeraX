# vim: set expandtab shiftwidth=4 softtabstop=4:

from chimerax.core.errors import UserError
from .utils import universe_to_atomic_structure, prep_coords
import numpy as np
import MDAnalysis as mda
import os

def read_structure(session, path, file_name, format_name=None, *, auto_style=True, coords=None, **kw):
    """
    Unified structure reader using MDAnalysis.
    Handles PSF, GRO, LAMMPS Data, etc.
    """


    # If an external coordinate file is provided (e.g. PSF + DCD)
    if coords:
        coords_path, _ = prep_coords(session, coords, path, format_name)
        load_args = [path, coords_path]
    else:
        load_args = [path]

    try:
        # Map ChimeraX format names to MDA format keywords if necessary
        # MDA usually auto-detects, but we can be explicit
        fmt_map = {
            "psf": "PSF",
            "gro": "GRO",
            "data": "DATA", # LAMMPS
            "pdb": "PDB"
        }
        mda_fmt = fmt_map.get(format_name, None)
        
        # Determine format kwarg
        kwargs = {}
        if mda_fmt:
            kwargs['format'] = mda_fmt
            
        # LAMMPS Data specific: MDA needs atom_style usually, but defaults to 'full'
        # If it fails, we might need to retry, but let's assume standard for now.
            
        universe = mda.Universe(path, coords, topology_format='PSF', format='LAMMPSDUMP', dt=1.0, in_memory=False)
        
    except ImportError:
        raise UserError("MDAnalysis is not installed. Please run 'pip install MDAnalysis' to use this bundle.")
    except Exception as e:
        session.logger.warning(f"MDAnalysis failed to load {path}: {e}")
        # Detailed error for LAMMPS which is common
        if format_name == "data":
             raise UserError(f"Failed to read LAMMPS data file. MDAnalysis error: {e}\n"
                             "Ensure the file has a standard header and Masses/Atoms sections.")
        raise UserError(f"Could not read file {file_name}: {e}")

    # Convert to ChimeraX structure
    try:
        name = os.path.basename(file_name)
        model = universe_to_atomic_structure(session, universe, name, auto_style=auto_style)
    except Exception as e:
         raise UserError(f"Failed to convert MDAnalysis topology to ChimeraX structure: {e}")

    # If external coords were loaded via MDA, we have one frame. 
    # If the file itself has coords (GRO), we have one frame.
    # PSF has 0 frames unless linked with coords.
    
    msg = f"Imported {model.num_atoms} atoms, {len(universe.trajectory)} frames."
    if len(universe.trajectory) > 0:
        #msg += f" Loaded frame 0 from {u.trajectory.filename}."
        
        for timestep in universe.trajectory:
            #session.logger.info(f"*** frame {timestep.frame} |positions| {len(timestep.positions)}")
            #session.logger.info(f"*** #{timestep.frame} {timestep.positions[:3]}")
            model.add_coordset(id=timestep.frame, xyz=timestep.positions.astype(np.float64))


    return [model], msg
