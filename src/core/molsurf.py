# vi: set expandtab shiftwidth=4 softtabstop=4:
"""
molsurf -- Compute molecular surfaces
=====================================
"""

from . import generic3d

class MolecularSurface(generic3d.Generic3DModel):
    pass

from . import cli, atomspec, color
_surface_desc = cli.CmdDesc(
    optional = [('atoms', atomspec.AtomSpecArg)],
    keyword = [('probeRadius', cli.FloatArg),
               ('gridSpacing', cli.FloatArg),
               ('color', color.ColorArg),
               ('transparency', cli.FloatArg),
               ('chains', cli.BoolArg)])

def surface_command(session, atoms = None, probeRadius = 1.4, gridSpacing = 0.5,
                    color = None, transparency = 0, chains = False):
    '''
    Compute and display a solvent excluded molecular surface for each molecule.
    '''
    surfs = []
    for name, xyz, r, place in atom_spec_spheres(atoms,session,chains):
        from . import surface
        va,na,ta = surface.ses_surface_geometry(xyz, r, probeRadius, gridSpacing)
        # Create surface model to show surface
        sname = '%s SES surface' % name
        rgba = surface_rgba(color, transparency, chains, name)
        surf = show_surface(sname, va, na, ta, rgba, place)
        session.models.add([surf])
        surfs.append(surf)
    return surfs

def atom_spec_spheres(atom_spec, session, chains = False):
    if atom_spec is None:
        s = []
        from .structure import StructureModel
        for m in session.models.list():
            if isinstance(m, StructureModel):
                a = m.mol_blob.atoms
                if chains:
                    for cname, ci in chain_indices(a):
                        xyz, r = a.coords, a.radii
                        s.append(('%s/%s'%(m.name,cname), xyz[ci], r[ci], m.position))
                else:
                    s.append((m.name, a.coords, a.radii, m.position))
    else:
        a = atom_spec.evaluate(session).atoms
        if a is None or len(a) == 0:
            raise cli.AnnotationError('No atoms specified by %s' % (str(atom_spec),))
        if chains:
            s = []
            for cname, ci in chain_indices(a):
                xyz, r = a.coords, a.radii
                s.append(('%s/%s'%(str(atom_spec),cname), xyz[ci], r[ci], None))
        else:
            s = [(str(atom_spec), a.coords, a.radii, None)]
        # TODO: Use correct position matrix for atoms
    return s

def chain_indices(atoms):
    import numpy
    atom_cids = numpy.array(atoms.residues.chain_ids)
    cids = numpy.unique(atom_cids)
    cid_masks = [(cid,(atom_cids == cid)) for cid in cids]
    return cid_masks

def surface_rgba(color, transparency, chains, cid):
    from .color import Color
    if chains and color is None:
        color = Color(chain_rgba(cid))
    if color is None:
        color = Color((.7,.7,.7,1))
    rgba8 = color.uint8x4()
    rgba8[3] = int(rgba8[3] * (100.0-transparency)/100.0)
    return rgba8

def chain_rgba(cid):
    from random import uniform, seed
    seed(str(cid))
    rgba = (uniform(.5,1),uniform(.5,1),uniform(.5,1),1)
    return rgba

def show_surface(name, va, na, ta, color = (180,180,180,255), place = None):

    surf = MolecularSurface(name)
    if not place is None:
        surf.position = place
    surf.geometry = va, ta
    surf.normals = na
    surf.color = color
    return surf

def register_surface_command():
    cli.register('surface', _surface_desc, surface_command)

_sasa_desc = cli.CmdDesc(
    optional = [('atoms', atomspec.AtomSpecArg)],
    keyword = [('probeRadius', cli.FloatArg),])

def sasa_command(session, atoms = None, probeRadius = 1.4):
    '''
    Compute solvent accessible surface area.
    Only the specified atoms are considered.
    '''
    log = session.logger
    for name, xyz, r, place in atom_spec_spheres(atoms,session):
        r += probeRadius
        from . import surface
        areas = surface.spheres_surface_area(xyz, r)
        area = areas.sum()
        msg = 'Solvent accessible area for %s = %.5g' % (name, area)
        log.info(msg)
        log.status(msg)

def register_sasa_command():
    cli.register('sasa', _sasa_desc, sasa_command)

_buriedarea_desc = cli.CmdDesc(
    required = [('atoms1', atomspec.AtomSpecArg), ('atoms2', atomspec.AtomSpecArg)],
    keyword = [('probeRadius', cli.FloatArg),])

def buriedarea_command(session, atoms1, atoms2, probeRadius = 1.4):
    '''
    Compute solvent accessible surface area.
    Only the specified atoms are considered.
    '''
    a1 = atoms1.evaluate(session).atoms
    a2 = atoms2.evaluate(session).atoms
    ni = len(a1.intersect(a2))
    if ni > 0:
        raise cli.AnnotationError('Two sets of atoms must be disjoint, got %d atoms in %s and %s'
                                  % (ni, str(atoms1), str(atoms2)))

    ba = buried_area(a1, a2, probeRadius)

    # Report result
    msg = 'Buried area between %s and %s = %.5g' % (str(atoms1), str(atoms2), ba)
    log = session.logger
    log.status(msg)
    msg += ('\n  area %s = %.5g, area %s = %.5g, area both = %.5g'
            % (str(atoms1), a1a, str(atoms2), a2a, a12a))
    log.info(msg)

def buried_area(a1, a2, probe_radius):
    from .surface import spheres_surface_area
    xyz1, r1 = atom_spheres(a1, probe_radius)
    a1a = spheres_surface_area(xyz1, r1).sum()
    xyz2, r2 = atom_spheres(a2, probe_radius)
    a2a = spheres_surface_area(xyz2, r2).sum()
    from numpy import concatenate
    xyz12, r12 = concatenate((xyz1,xyz2)), concatenate((r1,r2))
    a12a = spheres_surface_area(xyz12, r12).sum()
    ba = 0.5 * (a1a + a2a - a12a)

def atom_spheres(atoms, probe_radius = 1.4):
    xyz = atoms.coords
    r = atoms.radii.copy()
    r += probe_radius
    return xyz, r

def register_buriedarea_command():
    cli.register('buriedarea', _buriedarea_desc, buriedarea_command)
    cli.register('contact', _contact_desc, contact_command)

_contact_desc = cli.CmdDesc(
    optional = [('atoms', atomspec.AtomSpecArg),],
    keyword = [('probeRadius', cli.FloatArg),])

def contact_command(session, atoms = None, probeRadius = 1.4):
    '''
    Compute solvent accessible surface area.
    Only the specified atoms are considered.
    '''
    s = atom_spec_spheres(atoms, session, chains = True)
    areas, ba = buried_areas(s, probeRadius)

    names = [name for name,xyz,r,place in s]
    sn = dict(zip(names,short_chain_names(names)))

    # Report result
    msg = '%d buried areas: ' % len(ba) + ', '.join('%s %s %.0f' % (sn[n1],sn[n2],a) for n1,n2,a in ba)
    log = session.logger
    log.info(msg)
    log.status(msg)

    show_contact_graph(areas, ba, sn, session)

def short_chain_names(names):
    use_short_names = (len(set(n.split('/',1)[0] for n in names)) == 1)
    sn = tuple(n.split('/',1)[-1] for n in names) if use_short_names else names
    return sn

def buried_areas(s, probe_radius, min_area = 1):
    s = tuple((name, xyz, r + probe_radius, place) for name, xyz, r, place in s)

    areas = []
    from .surface import spheres_surface_area
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

# ------------------------------------------------------------------------------
#
from chimera.core.tools import ToolInstance
class Plot(ToolInstance):

    SIZE = (300, 300)

    def __init__(self, session, title = 'Plot'):

        super().__init__(session)

        from ..core.ui.tool_api import ToolWindow
        tw = ToolWindow(title, session, size=self.SIZE, destroy_hides=True)
        self.tool_window = tw
        parent = tw.ui_area

        from matplotlib import figure
        self.figure = f = figure.Figure(dpi=100, figsize=(2,2))
        from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as Canvas
        self.canvas = Canvas(parent, -1, f)

        import wx
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self.canvas,1,wx.EXPAND)
        parent.SetSizerAndFit(sizer)

        tw.manage(placement="right")

        self.axes = axes = f.gca()

    def show(self):
        self.tool_window.shown = True

    def hide(self):
        self.tool_window.shown = False

    #
    # Override ToolInstance methods
    #
    def display(self, b):
        """Show or hide map series user interface."""
        self.tool_window.shown = b

    #
    # Implement session.State methods if deriving from ToolInstance
    #
    def take_snapshot(self, session, flags):
        pass
    def restore_snapshot(self, phase, session, version, data):
        pass
    def reset_state(self):
        pass

def show_contact_graph(node_weights, edge_weights, short_names, session):

    # Create graph
    max_w = float(max(w for nm1,nm2,w in edge_weights))
    import networkx as nx
    G = nx.Graph()
    for name1, name2, w in edge_weights:
        G.add_edge(name1, name2, weight = w/max_w)

    # Layout nodes
    pos = nx.spring_layout(G) # positions for all nodes

    # Create matplotlib panel
    p = Plot(session, 'Chain Contacts')
    a = p.axes

    # Draw nodes
    from math import pow
    w = dict(node_weights)
    node_sizes = tuple(20*pow(w[n],1/3.0) for n in G)
    node_colors = tuple(chain_rgba(n) for n in G)
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, ax=a)

    # Draw edges
    esmall=[(u,v) for (u,v,d) in G.edges(data=True) if d['weight'] <=0.1]
    nx.draw_networkx_edges(G, pos, edgelist=esmall, width=2, style='dotted', ax=a)
    elarge=[(u,v) for (u,v,d) in G.edges(data=True) if d['weight'] >0.1]
    nx.draw_networkx_edges(G, pos, edgelist=elarge, width=3, ax=a)

    # Draw node labels
    nx.draw_networkx_labels(G, pos, labels=short_names, font_size=16, font_family='sans-serif', ax=a)

    # Hide axes and reduce border padding
    a.get_xaxis().set_visible(False)
    a.get_yaxis().set_visible(False)
    a.axis('tight')
    p.figure.tight_layout(pad = 0, w_pad = 0, h_pad = 0)
    p.show()
