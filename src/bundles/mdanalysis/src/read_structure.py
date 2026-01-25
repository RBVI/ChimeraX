# vim: set expandtab shiftwidth=4 softtabstop=4:

def read_structure(session, path, file_name, format_name=None, *, auto_style=True, coords=None, **kw):
    """
    Unified structure reader using MDAnalysis.
    Handles PSF, GRO, LAMMPS Data, etc.
    """
    from chimerax.core.errors import UserError
    from .utils import universe_to_atomic_structure, prep_coords
    import MDAnalysis as mda
    import os

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
        
        session.logger.info(f"*** ok 2\npath {path}\nfile_name {file_name}\ncoords {coords}\n")
        session.logger.info(f"*** load_args {load_args}\nkwargs {kwargs}\n")
            
        u = mda.Universe(path, coords, topology_format='PSF', format='LAMMPSDUMP', in_memory=True)
        
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
        s = universe_to_atomic_structure(session, u, name, auto_style=auto_style)
    except Exception as e:
         raise UserError(f"Failed to convert MDAnalysis topology to ChimeraX structure: {e}")

    # If external coords were loaded via MDA, we have one frame. 
    # If the file itself has coords (GRO), we have one frame.
    # PSF has 0 frames unless linked with coords.
    
    msg = f"Imported {s.num_atoms} atoms, {len(u.trajectory)} frames."
    if u.trajectory.n_frames > 0:
        msg += f" Loaded frame 0 from {u.trajectory.filename}."

    return [s], msg
