# vim: set expandtab ts=4 sw=4:

from chimerax.core.commands import CmdDesc, AtomsArg, FloatArg
contacts_desc = CmdDesc(
    optional = [('atoms', AtomsArg),],
    keyword = [('probe_radius', FloatArg),
               ('spring_constant', FloatArg)])

def contacts(session, atoms = None, probe_radius = 1.4, spring_constant = None):
    '''
    Compute buried solvent accessible surface areas between chains
    and show a 2-dimensional network graph depicting the contacts.

    Parameters
    ----------
    atoms : Atoms
    probe_radius : float
    '''
    s = atom_spheres(atoms, session)
    areas, ba = buried_areas(s, probe_radius)

    names = [name for name,xyz,r,color in s]
    sn = dict(zip(names,short_chain_names(names)))
    colors = {sn[name]:color for name,xyz,r,color in s}

    # Report result
    msg = '%d buried areas: ' % len(ba) + ', '.join('%s %s %.0f' % (sn[n1],sn[n2],a) for n1,n2,a in ba)
    log = session.logger
    log.info(msg)
    log.status(msg)

    if session.ui.is_gui:
        from . import gui
        gui.show_contact_graph(areas, ba, sn, colors, spring_constant, session)
    else:
        log.warning("unable to show graph without GUI")

def atom_spheres(atoms, session):
    if atoms is None:
        from chimerax.core.atomic import all_atoms
        atoms = all_atoms(session)
    if len(atoms) == 0:
        from chimerax.core.errors import UserError
        raise UserError('No atoms specified')
    from numpy import mean
    s = [('#%s/%s'%(m.id_string(),cid), catoms.scene_coords, catoms.radii, mean(catoms.colors,axis=0)/255)
         for m, cid, catoms in atoms.by_chain]
    return s

def short_chain_names(names):
    use_short_names = (len(set(n.split('/',1)[0] for n in names)) == 1)
    sn = tuple(n.split('/',1)[-1] for n in names) if use_short_names else names
    return sn

def buried_areas(s, probe_radius, min_area = 1):
    s = [(name, xyz, r + probe_radius) for name, xyz, r, color in s]
    s.sort(key = lambda v: len(v[1]), reverse = True)   # Biggest first for threading.
    
    # Compute area of each atom set.
    from chimerax.core.surface import spheres_surface_area
    from chimerax.core.threadq import apply_to_list
    def area(name, xyz, r):
        return (name, spheres_surface_area(xyz,r).sum())
    areas = apply_to_list(area, s)

    # Optimize buried area calculations using bounds of each atom set.
    naxes = 64
    from chimerax.core.geometry.sphere import sphere_points
    axes = sphere_points(naxes)
    from chimerax.core.geometry import sphere_axes_bounds
    bounds = [sphere_axes_bounds(xyz, r, axes) for name, xyz, r in s]

    # Compute buried areas between all pairs.
    buried = []
    n = len(s)
    pairs = []
    from chimerax.core.geometry import bounds_overlap
    for i in range(n):
        for j in range(i+1,n):
            if bounds_overlap(bounds[i], bounds[j], 0):
                pairs.append((i,j))

    def barea(i, j, s = s, bounds = bounds, axes = axes, probe_radius = probe_radius):
        n1, xyz1, r1 = s[i]
        n2, xyz2, r2 = s[j]
        ba = optimized_buried_area(xyz1, r1, bounds[i], xyz2, r2, bounds[j], axes, probe_radius)
        return (n1,n2,ba)
    bareas = apply_to_list(barea, pairs)
    buried = [(n1,n2,ba) for n1,n2,ba in bareas if ba >= min_area]
    buried.sort(key = lambda a: a[2], reverse = True)

    return areas, buried

# Consider only spheres in each set overlapping bounds of other set.
def optimized_buried_area(xyz1, r1, b1, xyz2, r2, b2, axes, probe_radius):

#    from chimerax.core.geometry import bounds_overlap, spheres_in_bounds
#    if not bounds_overlap(b1, b2, 0):
#        return 0

    from chimerax.core.geometry import spheres_in_bounds
    i1 = spheres_in_bounds(xyz1, r1, axes, b2, 0)
    i2 = spheres_in_bounds(xyz2, r2, axes, b1, 0)
    if len(i1) == 0 or len(i2) == 0:
        return 0

    xyz1, r1 = xyz1[i1], r1[i1]
    from chimerax.core.surface import spheres_surface_area
    a1 = spheres_surface_area(xyz1, r1).sum()
    xyz2, r2 = xyz2[i2], r2[i2]
    a2 = spheres_surface_area(xyz2, r2).sum()

    from numpy import concatenate
    xyz12, r12 = concatenate((xyz1,xyz2)), concatenate((r1,r2))
    a12 = spheres_surface_area(xyz12, r12).sum()
    ba = 0.5 * (a1 + a2 - a12)
    return ba

def buried_area(xyz1, r1, a1, xyz2, r2, a2):

    from numpy import concatenate
    xyz12, r12 = concatenate((xyz1,xyz2)), concatenate((r1,r2))
    from chimerax.core.surface import spheres_surface_area
    a12 = spheres_surface_area(xyz12, r12).sum()
    ba = 0.5 * (a1 + a2 - a12)
    return ba
