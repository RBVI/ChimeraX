# vi: set expandtab ts=4 sw=4:

from chimera.core.commands import CmdDesc, AtomSpecArg, FloatArg
contacts_desc = CmdDesc(
    optional = [('atoms', AtomSpecArg),],
    keyword = [('probeRadius', FloatArg),])

def contacts(session, atoms = None, probe_radius = 1.4):
    '''
    Compute buried solvent accessible surface areas between chains
    and show a 2-dimensional network graph depicting the contacts.

    Parameters
    ----------
    atoms : Atoms
    probe_radius : float
    '''
    from chimera.core import molsurf
    s = molsurf.atom_spec_spheres(atoms, session, chains = True)
    areas, ba = buried_areas(s, probe_radius)

    names = [name for name,xyz,r,place in s]
    sn = dict(zip(names,short_chain_names(names)))

    # Report result
    msg = '%d buried areas: ' % len(ba) + ', '.join('%s %s %.0f' % (sn[n1],sn[n2],a) for n1,n2,a in ba)
    log = session.logger
    log.info(msg)
    log.status(msg)

    from . import gui
    gui.show_contact_graph(areas, ba, sn, session)

def short_chain_names(names):
    use_short_names = (len(set(n.split('/',1)[0] for n in names)) == 1)
    sn = tuple(n.split('/',1)[-1] for n in names) if use_short_names else names
    return sn

def buried_areas(s, probe_radius, min_area = 1):
    # TODO: Use place matrices
    s = [(name, xyz, r + probe_radius, place) for name, xyz, r, place in s]
    s.sort(key = lambda v: len(v[1]), reverse = True)   # Biggest first for threading.
    
    # Compute area of each atom set.
    from chimera.core.surface import spheres_surface_area
    from chimera.core.threadq import apply_to_list
    def area(name, xyz, r, place):
        return (name, spheres_surface_area(xyz,r).sum())
    areas = apply_to_list(area, s)

    # Optimize buried area calculations using bounds of each atom set.
    naxes = 64
    from chimera.core.geometry.sphere import sphere_points
    axes = sphere_points(naxes)
    from chimera.core.geometry import sphere_bounds
    bounds = [sphere_bounds(xyz, r, axes) for name, xyz, r, place in s]

    # Compute buried areas between all pairs.
    buried = []
    n = len(s)
    pairs = []
    from chimera.core.geometry import bounds_overlap
    for i in range(n):
        for j in range(i+1,n):
            if bounds_overlap(bounds[i], bounds[j], 0):
                pairs.append((i,j))

    def barea(i, j, s = s, bounds = bounds, axes = axes, probe_radius = probe_radius):
        n1, xyz1, r1, p1 = s[i]
        n2, xyz2, r2, p2 = s[j]
        ba = optimized_buried_area(xyz1, r1, bounds[i], xyz2, r2, bounds[j], axes, probe_radius)
        return (n1,n2,ba)
    bareas = apply_to_list(barea, pairs)
    buried = [(n1,n2,ba) for n1,n2,ba in bareas if ba >= min_area]
    buried.sort(key = lambda a: a[2], reverse = True)

    return areas, buried

# Consider only spheres in each set overlapping bounds of other set.
def optimized_buried_area(xyz1, r1, b1, xyz2, r2, b2, axes, probe_radius):

#    from chimera.core.geometry import bounds_overlap, spheres_in_bounds
#    if not bounds_overlap(b1, b2, 0):
#        return 0

    from chimera.core.geometry import spheres_in_bounds
    i1 = spheres_in_bounds(xyz1, r1, axes, b2, 0)
    i2 = spheres_in_bounds(xyz2, r2, axes, b1, 0)
    if len(i1) == 0 or len(i2) == 0:
        return 0

    xyz1, r1 = xyz1[i1], r1[i1]
    from chimera.core.surface import spheres_surface_area
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
    from chimera.core.surface import spheres_surface_area
    a12 = spheres_surface_area(xyz12, r12).sum()
    ba = 0.5 * (a1 + a2 - a12)
    return ba
