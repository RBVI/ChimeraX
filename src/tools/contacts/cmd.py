# vim: set expandtab ts=4 sw=4:

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
    sg = chain_spheres(atoms, session)
    ba = buried_areas(sg, probe_radius)

    for g,sname in zip(sg,short_chain_names([g.name for g in sg])):
        g.short_name = sname

    # Report result
    msg = '%d buried areas: ' % len(ba) + ', '.join('%s %s %.0f' % (g1.short_name,g2.short_name,a) for g1,g2,a in ba)
    log = session.logger
    log.info(msg)
    log.status(msg)

    if session.ui.is_gui:
        def graph_clicked(sphere_groups, event, all_sphere_groups = sg, session=session):
            sg = sphere_groups
            if event.key == 'shift':
                session.selection.clear()
                for g in sg:
                    for m, matoms in g.atoms.by_structure:
                        m.select_atoms(matoms)
            else:
                n = len(sg)
                if n == 0:
                    for h in all_sphere_groups:
                        h.atoms.displays = True
                elif n == 1:
                    g = sg[0]
                    ng = neigbhors(g, ba)
                    ng.add(g)
                    for h in all_sphere_groups:
                        h.atoms.displays = (h in ng)
                else:
                    # Edge clicked, g = pair of sphere groups
                    gset = set(sg)
                    for h in all_sphere_groups:
                        h.atoms.displays = (h in gset)
                    
#            print ('event button', event.button, 'key', event.key, 'step', event.step)
        from . import gui
        gui.ContactPlot(session, sg, ba, spring_constant, graph_clicked)
    else:
        log.warning("unable to show graph without GUI")



class SphereGroup:
    def __init__(self, name, atoms):
        self.name = name
        self.atoms = atoms
        self.centers = atoms.scene_coords
        self.radii = atoms.radii
        from numpy import mean
        self.color = mean(atoms.colors,axis=0)/255.0
        self.area = None

    def shown(self):
        a = self.atoms
        return a.displays.any() or a.residues.ribbon_displays.any()
        
def chain_spheres(atoms, session):
    if atoms is None:
        from chimerax.core.atomic import all_atoms
        atoms = all_atoms(session)
    if len(atoms) == 0:
        from chimerax.core.errors import UserError
        raise UserError('No atoms specified')
    from numpy import mean
    s = [SphereGroup('#%s/%s'%(m.id_string(),cid), catoms)
         for m, cid, catoms in atoms.by_chain]
    return s

def short_chain_names(names):
    use_short_names = (len(set(n.split('/',1)[0] for n in names)) == 1)
    sn = tuple(n.split('/',1)[-1] for n in names) if use_short_names else names
    return sn

def buried_areas(sphere_groups, probe_radius, min_area = 1):
    s = [(g, g.radii + probe_radius) for g in sphere_groups]
    s.sort(key = lambda v: len(v[1]), reverse = True)   # Biggest first for threading.
    
    # Compute area of each atom set.
    from chimerax.core.surface import spheres_surface_area
    from chimerax.core.threadq import apply_to_list
    def area(g, r):
        g.area = spheres_surface_area(g.centers,r).sum()
    apply_to_list(area, s)

    # Optimize buried area calculations using bounds of each atom set.
    naxes = 64
    from chimerax.core.geometry.sphere import sphere_points
    axes = sphere_points(naxes)
    from chimerax.core.geometry import sphere_axes_bounds
    bounds = [sphere_axes_bounds(g.centers, r, axes) for g, r in s]

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
        g1, r1 = s[i]
        g2, r2 = s[j]
        ba = optimized_buried_area(g1.centers, r1, bounds[i], g2.centers, r2, bounds[j], axes, probe_radius)
        return (g1,g2,ba)
    bareas = apply_to_list(barea, pairs)
    buried = [(g1,g2,ba) for g1,g2,ba in bareas if ba >= min_area]
    buried.sort(key = lambda a: a[2], reverse = True)

    return buried

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

def neigbhors(g, buried_areas):
    n = set()
    for g1,g2,w in buried_areas:
        if w > 0:
            if g1 is g:
                n.add(g2)
            elif g2 is g:
                n.add(g1)
    return n
