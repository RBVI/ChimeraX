def show_surface(name, va, na, ta, session, color = (.7,.7,.7,1), place = None):

    from ..models import Model
    surf = Model(name)
    if not place is None:
        surf.position = place
    surf.geometry = va, ta
    surf.normals = na
    surf.color = color
    session.add_model(surf)
    return surf

def surface_command(cmdname, args, session):

    from ..commands.parse import atoms_arg, float_arg, no_arg, parse_arguments
    req_args = (('atoms', atoms_arg),)
    opt_args = ()
    kw_args = (('probeRadius', float_arg),
               ('gridSpacing', float_arg),
               ('waters', no_arg),)

    kw = parse_arguments(cmdname, args, session, req_args, opt_args, kw_args)
    kw['session'] = session
    surface(**kw)

def surface(atoms, session, probeRadius = 1.4, gridSpacing = 0.5, waters = False):
    '''
    Compute and display a solvent excluded molecular surfaces for specified atoms.
    If waters is false then water residues (residue name HOH) are removed from
    the atom set before computing the surface.
    '''
    if not waters:
        atoms = atoms.exclude_water()
    xyz = atoms.coordinates()           # Scene coordinates
    r = atoms.radii()
    from .. import surface
    va,na,ta = surface.ses_surface_geometry(xyz, r, probeRadius, gridSpacing)

    # Create surface model to show surface
    m0 = atoms.molecules()[0]
    p = m0.position
    if not p.is_identity(tolerance = 0):
        p.inverse().move(va)    # Move to model coordinates.
        
    name = '%s SES surface' % m0.name
    surf = show_surface(name, va, na, ta, session, color = (180,205,128,255), place = p)
    surf.ses_atoms = atoms

    return surf
