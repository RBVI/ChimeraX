# vi: set expandtab ts=4 sw=4:

from chimera.core import cli, atomspec
contact_desc = cli.CmdDesc(
    optional = [('atoms', atomspec.AtomSpecArg),],
    keyword = [('probeRadius', cli.FloatArg),])

def contact_command(session, atoms = None, probeRadius = 1.4):
    '''
    Compute solvent accessible surface area.
    Only the specified atoms are considered.
    '''
    from chimera.core import molsurf
    s = molsurf.atom_spec_spheres(atoms, session, chains = True)
    areas, ba = buried_areas(s, probeRadius)

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
    s = tuple((name, xyz, r + probe_radius, place) for name, xyz, r, place in s)

    areas = []
    from chimera.core.surface import spheres_surface_area
    for name, xyz, r, place in s:
        areas.append(spheres_surface_area(xyz, r).sum())

    # TODO: Use place matrices
    buried = []
    n = len(s)
    for i in range(n):
        n1, xyz1, r1, p1 = s[i]
        for j in range(i+1,n):
            n2, xyz2, r2, p2 = s[j]
            from numpy import concatenate
            xyz12, r12 = concatenate((xyz1,xyz2)), concatenate((r1,r2))
            a12 = spheres_surface_area(xyz12, r12).sum()
            ba = 0.5 * (areas[i] + areas[j] - a12)
            if ba >= min_area:
                buried.append((n1, n2, ba))
    buried.sort(key = lambda a: a[2], reverse = True)

    ca = zip(tuple(name for name,xyz,r,place in s), areas)
    return ca, buried
