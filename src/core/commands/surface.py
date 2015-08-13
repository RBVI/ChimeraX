# vi: set expandtab shiftwidth=4 softtabstop=4:

# -------------------------------------------------------------------------------------
#
def surface_command(session, atoms = None, enclose = None, include = None,
                    probe_radius = 1.4, grid_spacing = None, resolution = None, level = None,
                    color = None, transparency = None, visible_patches = None,
                    sharp_boundaries = None,
                    nthread = None, replace = True, show = False, hide = False, close = False):
    '''
    Compute and display solvent excluded molecular surfaces.
    '''
    atoms = check_atoms(atoms, session) # Warn if no atoms specifed

    from ..molsurf import close_surfaces, show_surfaces, hide_surfaces, remove_solvent_ligands_ions
    from ..molsurf import surface_rgba, MolecularSurface, update_color, surfaces_overlapping_atoms

    if close:
        close_surfaces(atoms, session.models)
        return []

    # Show surface patches for existing surfaces.
    if show:
        return show_surfaces(atoms, session.models)

    # Hide surfaces or patches of surface for specified atoms.
    if hide:
        return hide_surfaces(atoms, session.models)

    if replace:
        all_surfs = dict((s.atoms.hash(), s) for s in session.models.list(type = MolecularSurface))
    else:
        all_surfs = {}

    if grid_spacing is None:
        grid_spacing = 0.5 if resolution is None else 0.1 * resolution

    if sharp_boundaries is None:
        sharp_boundaries = True if resolution is None else False

    surfs = []
    new_surfs = []
    if enclose is None:
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
                name = '%s_%s SES surface' % (m.name, chain_id)
                rgba = surface_rgba(color, transparency, chain_id)
                s = MolecularSurface(enclose_atoms, show_atoms,
                                     probe_radius, grid_spacing, resolution, level,
                                     name, rgba, visible_patches, sharp_boundaries)
                new_surfs.append((s,m))
            else:
                s.new_parameters(show_atoms, probe_radius, grid_spacing,
                                 resolution, level, visible_patches, sharp_boundaries)
                update_color(s, color, transparency)
            surfs.append(s)
    else:
        enclose_atoms, eall_small = remove_solvent_ligands_ions(enclose, include)
        show_atoms = enclose_atoms if atoms is None else atoms.intersect(enclose_atoms)
        s = all_surfs.get(enclose_atoms.hash())
        if s is None:
            mols = enclose.unique_structures
            parent = mols[0] if len(mols) == 1 else session.models.drawing
            name = 'Surface %s' % enclose.spec
            rgba = surface_rgba(color, transparency)
            s = MolecularSurface(enclose_atoms, show_atoms,
                                 probe_radius, grid_spacing, resolution, level,
                                 name, rgba, visible_patches, sharp_boundaries)
            new_surfs.append((s,parent))
        else:
            s.new_parameters(show_atoms, probe_radius, grid_spacing,
                             resolution, level, visible_patches, sharp_boundaries)
            update_color(s, color, transparency)
        surfs.append(s)

    # Close overlapping surfaces.
    if replace:
        other_surfs = set(all_surfs.values()) - set(surfs)
        from ..molecule import concatenate
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

def register_surface_command():
    from . import CmdDesc, register, AtomsArg, FloatArg, IntArg, ColorArg, BoolArg, NoArg
    _surface_desc = CmdDesc(
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
                   ('replace', BoolArg),
                   ('show', NoArg),
                   ('hide', NoArg),
                   ('close', NoArg)],
        synopsis = 'create molecular surface')
    register('surface', _surface_desc, surface_command)

def check_atoms(atoms, session):
    if atoms is None:
        from ..structure import all_atoms
        atoms = all_atoms(session)
        if len(atoms) == 0:
            from . import AnnotationError
            raise AnnotationError('No atomic models open.')
        atoms.spec = 'all atoms'
    elif len(atoms) == 0:
        from . import AnnotationError
        raise AnnotationError('No atoms specified by %s' % (atoms.spec,))
    return atoms
