# vim: set expandtab shiftwidth=4 softtabstop=4:

def read_coords(session, file_name, model, format_name, *, replace=True, start=1, step=1, end=None):
    """
    Unified trajectory reader using MDAnalysis.
    """
    from chimerax.core.errors import UserError, LimitationError
    import MDAnalysis as mda
    import numpy as np

    # Adjust 1-based start to 0-based
    start_idx = start - 1
    step_val = step
    
    try:
        # Create a Universe using the ChimeraX model as the "topology" source of truth regarding atom count.
        # Since we can't easily convert ChimeraX -> MDA Topology object in memory without temporary files,
        # we use the 'empty' Universe approach for pure trajectory formats (DCD, XTC, TRR)
        # where the topology file isn't strictly required if atom counts match.
        
        # However, MDA usually requires a topology to load XTC/DCD. 
        # We can construct an "Anonymous" topology with the correct number of atoms.
        
        n_atoms = model.num_atoms
        
        # Create empty universe with correct atom count
        u = mda.Universe.empty(n_atoms, trajectory=True)
        
        # Map format names
        fmt_map = {
            "dcd": "DCD",
            "xtc": "XTC",
            "trr": "TRR",
            "amber": "NCDF", # Amber NetCDF
            "dump": "LAMMPS", # LAMMPS dump
        }
        mda_fmt = fmt_map.get(format_name, format_name)

        # Load the trajectory into the empty universe
        try:
            u.load_new(file_name, format=mda_fmt)
        except Exception as e:
            # Fallback: sometimes loading fails if format isn't guessed. 
            # Try letting MDA guess from extension if map failed
            u.load_new(file_name)
            
    except ImportError:
        raise UserError("MDAnalysis is not installed. Please run 'pip install MDAnalysis'.")
    except Exception as e:
        raise UserError(f"Failed to read trajectory {file_name} with MDAnalysis: {e}")

    # Verify atom counts
    if u.trajectory.n_atoms != model.num_atoms:
        raise UserError(f"Trajectory has {u.trajectory.n_atoms} atoms, but structure has {model.num_atoms}.")

    # Handle range
    n_frames_total = u.trajectory.n_frames
    if end is None:
        end_idx = n_frames_total
    else:
        end_idx = end

    if end_idx > n_frames_total:
        session.logger.warning(f"Requested end frame {end_idx} exceeds total frames {n_frames_total}. Truncating.")
        end_idx = n_frames_total

    # Prepare ChimeraX coordinate sets
    if replace:
        model.remove_coordsets()
        base_id = 1
    else:
        base_id = max(model.coordset_ids) + 1 if model.coordset_ids else 1

    # Iterate and load
    # MDA slicing u.trajectory[start:stop:step] returns an iterator
    
    count = 0
    try:
        for ts in u.trajectory[start_idx:end_idx:step_val]:
            # ts.positions is a numpy array of shape (N, 3) in Angstroms
            # ChimeraX expects Angstroms.
            # GROMACS XTC/TRR are in nm in file, but MDA converts to Angstroms automatically.
            
            coords = ts.positions.astype(np.float64)
            model.add_coordset(base_id + count, coords)
            count += 1
            
            # Optional: Status update for large files
            if count % 100 == 0:
                session.logger.status(f"Reading frame {count}...", blank_after=0)
                
    except Exception as e:
        raise UserError(f"Error reading frames from {file_name}: {e}")

    session.logger.status(f"Finished reading {count} frames from {file_name}.")
    
    # Set active coordset
    model.active_coordset_id = base_id
    
    return count
