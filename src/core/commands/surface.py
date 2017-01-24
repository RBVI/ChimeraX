# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2016 Regents of the University of California.
# All rights reserved.  This software provided pursuant to a
# license agreement containing restrictions on its disclosure,
# duplication and use.  For details see:
# http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html
# This notice must be embedded in or attached to all copies,
# including partial copies, of the software or any revisions
# or derivations thereof.
# === UCSF ChimeraX Copyright ===

# -------------------------------------------------------------------------------------
#
def surface(session, atoms = None, enclose = None, include = None,
            probe_radius = 1.4, grid_spacing = None, resolution = None, level = None,
            color = None, transparency = None, visible_patches = None,
            sharp_boundaries = None, nthread = None, replace = True):
    '''
    Compute and display solvent excluded molecular surfaces.

    Parameters
    ----------
    atoms : Atoms
      Atoms controlling which surface patch is shown.
      Surfaces are computed for each chain these atoms belong to and patches
      of these surfaces near the specifed atoms are shown.  Solvent, ligands
      and ions are excluded from each chain surface.
    enclose : Atoms
      Make a surface enclosing exactly these specified atoms excluding
      solvent, ligands and ions
    include : Atoms
      Solvent, ligands or ions to include in the surface.
    probe_radius : float
      Radius of probe sphere rolled over atoms to produce surface.
      Only used for solvent excluded surfaces.
    grid_spacing : float
      Surface is computed on 3-dimensional grid with this spacing
      between grid points along each axis.
    resolution : float
      Specifying a resolution value (Angstroms) causes the surface calculation
      to use a contour surface of a 3-d grid of a sum of Gaussians one centered
      at each atom instead of a solvent excluded surface.  See the molmap command
      for details.
    level : float
      Contour level for Gaussian surface in atomic number units.  Each Gaussian has
      height scaled by the atom atomic number.
    color : Color
      Colors surfaces using this color.
    transparency : float
      Percentage transparency for surfaces.
    visible_patches : int
      Maximum number of connected surface pieces per chain to show.
    sharp_boundaries : bool
      Make the surface triangulation have edges exactly between atoms
      so per-atom surface colors and surface patches have smoother edges.
    nthread : int
      Number of CPU threads to use in computing surfaces.
    replace : bool
      Whether to replace an existing surface for the same atoms or make a copy.
    '''

    from ..atomic.molsurf import MolecularSurface, remove_solvent_ligands_ions
    from ..atomic.molsurf import surface_rgba, update_color, surfaces_overlapping_atoms

    if replace:
        all_surfs = dict((s.atoms.hash(), s) for s in session.models.list(type = MolecularSurface))
    else:
        all_surfs = {}

    if grid_spacing is None:
        grid = 0.5 if resolution is None else 0.1 * resolution
    else:
        grid = grid_spacing

    if sharp_boundaries is None:
        sharp = True if resolution is None else False
    else:
        sharp = sharp_boundaries

    surfs = []
    new_surfs = []
    if enclose is None:
        atoms = check_atoms(atoms, session) # Warn if no atoms specifed
        atoms, all_small = remove_solvent_ligands_ions(atoms, include)
        for m, chain_id, show_atoms in atoms.by_chain:
            if all_small:
                enclose_atoms = atoms.filter(atoms.chain_ids == chain_id)
            else:
                matoms = m.atoms
                chain_atoms = matoms.filter(matoms.chain_ids == chain_id)
                enclose_atoms = remove_solvent_ligands_ions(chain_atoms, include)[0]
            s = all_surfs.get(enclose_atoms.hash())
            if s is None:
                stype = 'SES' if resolution is None else 'Gaussian'
                name = '%s_%s %s surface' % (m.name, chain_id, stype)
                rgba = surface_rgba(color, transparency, chain_id)
                s = MolecularSurface(session, enclose_atoms, show_atoms,
                                     probe_radius, grid, resolution, level,
                                     name, rgba, visible_patches, sharp)
                new_surfs.append((s,m))
            else:
                s.new_parameters(show_atoms, probe_radius, grid_spacing,
                                 resolution, level, visible_patches, sharp_boundaries)
                update_color(s, color, transparency)
            surfs.append(s)
    else:
        enclose_atoms, eall_small = remove_solvent_ligands_ions(enclose, include)
        if len(enclose_atoms) == 0:
            from ..errors import UserError
            raise UserError('No atoms specified by %s' % (enclose.spec,))
        show_atoms = enclose_atoms if atoms is None else atoms.intersect(enclose_atoms)
        s = all_surfs.get(enclose_atoms.hash())
        if s is None:
            mols = enclose.unique_structures
            parent = mols[0] if len(mols) == 1 else None
            name = 'Surface %s' % enclose.spec
            rgba = surface_rgba(color, transparency)
            s = MolecularSurface(session, enclose_atoms, show_atoms,
                                 probe_radius, grid, resolution, level,
                                 name, rgba, visible_patches, sharp)
            new_surfs.append((s,parent))
        else:
            s.new_parameters(show_atoms, probe_radius, grid_spacing,
                             resolution, level, visible_patches, sharp_boundaries)
            update_color(s, color, transparency)
        surfs.append(s)

    # Close overlapping surfaces.
    if replace:
        other_surfs = set(all_surfs.values()) - set(surfs)
        from ..atomic import concatenate
        surf_atoms = concatenate([s.atoms for s in surfs])
        osurfs = surfaces_overlapping_atoms(other_surfs, surf_atoms)
        if osurfs:
            session.models.close(osurfs)

    # Compute surfaces using multiple threads
    args = [(s,) for s in surfs]
    args.sort(key = lambda s: s[0].atom_count, reverse = True)      # Largest first for load balancing
    from .. import threadq
    threadq.apply_to_list(lambda s: s.calculate_surface_geometry(), args, nthread)
#    for s in surfs:
#        s.calculate_surface_geometry()
    # TODO: Any Python error in the threaded call causes a crash when it tries
    #       to write an error message to the log, not in the main thread.

    if not resolution is None and level is None:
        log = session.logger
        log.info('\n'.join('%s contour level %.1f' % (s.name, s.gaussian_level)
                           for s in surfs))
            
    # Add new surfaces to open models list.
    for s, parent in new_surfs:
        session.models.add([s], parent = parent)

    # Make sure replaced surfaces are displayed.
    for s in surfs:
        s.display = True

    return surfs

# -------------------------------------------------------------------------------------
#
def surface_show(session, atoms = None):
    '''
    Show surface patches for atoms of existing surfaces.

    Parameters
    ----------
    atoms : Atoms
      Show surface patches for existing surfaces.
    '''
    from ..atomic.molsurf import show_surfaces
    atoms = check_atoms(atoms, session) # Warn if no atoms specifed
    return show_surfaces(atoms, session.models)

# -------------------------------------------------------------------------------------
#
def surface_hide(session, atoms = None):
    '''
    Hide patches of existing surfaces for specified atoms.

    Parameters
    ----------
    atoms : Atoms
    '''
    from ..atomic.molsurf import hide_surfaces
    atoms = check_atoms(atoms, session) # Warn if no atoms specifed
    return hide_surfaces(atoms, session.models)

# -------------------------------------------------------------------------------------
#
def surface_close(session, objects = None):
    '''
    Close molecular surfaces.

    Parameters
    ----------
    objects : Objects
      Close specified molecular surfaces including any surfaces computed using specified atoms.
    '''
    from ..atomic.molsurf import close_surfaces, MolecularSurface
    if objects is None:
        surfs = session.models.list(type = MolecularSurface)
    else:
        surfs = [s for s in objects.models if isinstance(s, MolecularSurface)]
    close_surfaces(surfs, session.models)

    if objects:
        close_surfaces(objects.atoms, session.models)

# -------------------------------------------------------------------------------------
#
def unsurface(session, atoms = None):
    '''
    Hide surface for specified atoms.  Same as command "surface <spec> hide".
    '''
    surface_hide(session, atoms)

# -------------------------------------------------------------------------------------
#
def register_command(session):
    from . import CmdDesc, register, ObjectsArg, AtomsArg
    from . import FloatArg, IntArg, ColorArg, BoolArg, NoArg, create_alias
    surface_desc = CmdDesc(
        optional = [('atoms', AtomsArg)],
        keyword = [('enclose', AtomsArg),
                   ('include', AtomsArg),
                   ('probe_radius', FloatArg),
                   ('grid_spacing', FloatArg),
                   ('resolution', FloatArg),
                   ('level', FloatArg),
                   ('color', ColorArg),
                   ('transparency', FloatArg),
                   ('visible_patches', IntArg),
                   ('sharp_boundaries', BoolArg),
                   ('nthread', IntArg),
                   ('replace', BoolArg)],
        synopsis = 'create molecular surface')
    register('surface', surface_desc, surface)

    show_desc = CmdDesc(
        optional = [('atoms', AtomsArg)],
        synopsis = 'Show patches of molecular surfaces')
    register('surface show', show_desc, surface_show)

    hide_desc = CmdDesc(
        optional = [('atoms', AtomsArg)],
        synopsis = 'Hide patches of molecular surfaces')
    register('surface hide', hide_desc, surface_hide)

    close_desc = CmdDesc(
        optional = [('objects', ObjectsArg)],
        synopsis = 'close molecular surfaces')
    register('surface close', close_desc, surface_close)
    
    unsurface_desc = CmdDesc(
        optional = [('atoms', AtomsArg)],
        synopsis = 'hide molecular surface')
    register('~surface', unsurface_desc, unsurface)

def check_atoms(atoms, session):
    if atoms is None:
        from ..atomic import all_atoms
        atoms = all_atoms(session)
        if len(atoms) == 0:
            from ..errors import UserError
            raise UserError('No atomic models open.')
        atoms.spec = 'all atoms'
    elif len(atoms) == 0:
        msg = 'No atoms specified'
        if hasattr(atoms, 'spec'):
            msg += ' by %s' % atoms.spec
        from ..errors import UserError
        raise UserError(msg)
    return atoms
