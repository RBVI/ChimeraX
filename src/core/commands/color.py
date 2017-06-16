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

_SpecialColors = ["byatom", "byelement", "byhetero", "bychain", "bypolymer", "bymodel",
                  "fromatoms", "random"]

_SequentialLevels = ["residues", "chains", "polymers", "structures"]
# More possible sequential levels: "helix", "helices", "strands", "SSEs", "volmodels", "allmodels"

def color(session, objects, color=None, what=None,
          target=None, transparency=None,
          sequential=None, palette=None, halfbond=None,
          map=None, range=None, offset=0, zone=None, distance=2):
    """Color atoms, ribbons, surfaces, ....

    Parameters
    ----------
    objects : Objects
      Which objects to color.
    color : Color
      Color can be a standard color name or "byatom", "byelement", "byhetero", "bychain", "bypolymer", "bymodel".
    what :  'atoms', 'cartoons', 'ribbons', 'surfaces', 'bonds', 'pseudobonds' or None
      What to color. Everything is colored if option is not specified.
    target : string
      Alternative to the "what" option for specifying what to color.
      Characters indicating what to color, a = atoms, c = cartoon, r = cartoon, s = surfaces,
      l = labels, b = bonds, p = pseudobonds, d = distances.
      Everything is colored if no target is specified.
    transparency : float
      Percent transparency to use.  If not specified current transparency is preserved.
    sequential : "residues", "chains", "polymers", "structures"
      Assigns each object a color from a color map.
    palette : :class:`.Colormap`
      Color map to use with sequential coloring.
    halfbond : bool
      Whether to color each half of a bond to match the connected atoms.
      If halfbond is false the bond is given the single color assigned to the bond.
    map : Volume
      Color specified surfaces by sampling from this density map using palette, range, and offset options.
    range : 2 comma-separated floats or "full"
      Specifies the range of map values used for sampling from a palette.
    offset : float
      Displacement distance along surface normals for sampling map when using map option.  Default 0.
    zone : Atoms
      Color surfaces to match closest atom within specified zone distance.
    distance : float
      Zone distance used with zone option.
    """
    if objects is None:
        from . import all_objects
        objects = all_objects(session)
    atoms = objects.atoms
    if color == "byhetero":
        atoms = atoms.filter(atoms.element_numbers != 6)

    default_target = (target is None and what is None)
    if default_target:
        target = 'acslbd'
    if target and 'r' in target:
        target += 'c'

    if what is not None:
        what_target = {'atoms':'a', 'cartoons':'c', 'ribbons':'c',
                       'surfaces':'s', 'bonds':'b', 'pseudobonds':'p'}
        if target is None:
            target = ''
        target += what_target[what]

    # Decide whether to set or preserve transparency
    opacity = None
    if transparency is not None:
        opacity = min(255, max(0, int(2.56 * (100 - transparency))))
    if getattr(color, 'explicit_transparency', False):
        opacity = color.uint8x4()[3]

    if halfbond is not None and atoms is not None:
        bonds = atoms.intra_bonds
        if len(bonds) > 0:
            bonds.halfbonds = halfbond

    if sequential is not None:
        try:
            f = _SequentialColor[sequential]
        except KeyError:
            from ..errors import UserError
            raise UserError("sequential \"%s\" not implemented yet"
                            % sequential)
        else:
            f(session, objects, palette, opacity, target)
            return

    if zone is not None:
        from ..atomic import MolecularSurface, Structure
        slist = [m for m in objects.models
                 if not m.empty_drawing() and not isinstance(m, (Structure, MolecularSurface))]
        for m in objects.models:
            if hasattr(m, 'surface_drawings_for_vertex_coloring'):
                slist.extend(m.surface_drawings_for_vertex_coloring())
        bonds = None
        auto_update = False
        from ..surface.colorzone import points_and_colors, color_zone
        for s in slist:
            points, colors = points_and_colors(zone, bonds)
            s.scene_position.inverse().move(points)	# Transform points to surface coordinates
            color_zone(s, points, colors, distance, auto_update)

    what = []

    bgcolor = session.main_view.background_color

    if 'a' in target:
        # atoms/bonds
        if atoms is not None and color is not None:
            _set_atom_colors(atoms, color, opacity, bgcolor)
            what.append('%d atoms' % len(atoms))

    if 'l' in target:
        if not default_target:
            session.logger.warning('Label colors not supported yet')

    if 's' in target and (color is not None or map is not None):
        from ..atomic import MolecularSurface, concatenate, Structure
        msatoms = [m.atoms for m in objects.models
                   if isinstance(m, MolecularSurface) and not m.atoms.intersects(atoms)]
        satoms = concatenate(msatoms + [atoms]) if msatoms else atoms
        if color == "byhetero":
            satoms = satoms.filter(satoms.element_numbers != 6)
        ns = _set_surface_colors(session, satoms, color, opacity, bgcolor, map, palette, range, offset)
        # Handle non-molecular surfaces like density maps
        if color not in _SpecialColors:
            mlist = [m for m in objects.models if not isinstance(m, (Structure, MolecularSurface))]
            for m in mlist:
                _set_model_colors(session, m, color, map, opacity, palette, range, offset)
            ns += len(mlist)
        what.append('%d surfaces' % ns)

    if 'c' in target and color is not None:
        residues = atoms.unique_residues
        _set_ribbon_colors(residues, color, opacity, bgcolor)
        what.append('%d residues' % len(residues))

    if 'b' in target and color is not None:
        if atoms is not None:
            bonds = atoms.intra_bonds
            if len(bonds) > 0:
                if color not in _SpecialColors:
                    bonds.colors = color.uint8x4()
                    what.append('%d bonds' % len(bonds))

    if 'p' in target:
        if atoms is not None:
            from .. import atomic
            bonds = atomic.interatom_pseudobonds(atoms)
            if len(bonds) > 0:
                if color not in _SpecialColors and color is not None:
                    bonds.colors = color.uint8x4()
                what.append('%d pseudobonds' % len(bonds))

    if 'd' in target:
        if not default_target:
            session.logger.warning('Distances colors not supported yet')

    if not what:
        what.append('nothing')

    from . import cli
    session.logger.status('Colored %s' % cli.commas(what, ' and'))


def _computed_atom_colors(atoms, color, opacity, bgcolor):
    if color in ("byatom", "byelement", "byhetero"):
        c = _element_colors(atoms, opacity)
    elif color == "bychain":
        from ..atomic.colors import chain_colors
        c = chain_colors(atoms.residues.chain_ids)
        c[:, 3] = atoms.colors[:, 3] if opacity is None else opacity
    elif color == "bypolymer":
        from ..atomic.colors import polymer_colors
        c = atoms.colors.copy()
        sc,amask = polymer_colors(atoms.residues)
        c[amask,:] = sc[amask,:]
        c[amask, 3] = atoms.colors[amask, 3] if opacity is None else opacity
    elif color == "bymodel":
        c = atoms.colors.copy()
        for m, matoms in atoms.by_structure:
            color = m.initial_color(bgcolor).uint8x4()
            mi = atoms.mask(matoms)
            c[mi, :3] = color[:3]
            if opacity is not None:
                c[mi, 3] = opacity
    elif color == "random":
        from numpy import random, uint8
        c = random.randint(0, 255, (len(atoms), 4)).astype(uint8)
        c[:, 3] = 255   # Opaque
    else:
        # Other "colors" do not apply to atoms
        c = None
    return c


def _element_colors(atoms, opacity=None):
    from ..atomic.colors import element_colors
    c = element_colors(atoms.element_numbers)
    c[:, 3] = atoms.colors[:, 3] if opacity is None else opacity
    return c


def _set_atom_colors(atoms, color, opacity, bgcolor=None):
    if color in _SpecialColors:
        c = _computed_atom_colors(atoms, color, opacity, bgcolor)
        if c is not None:
            atoms.colors = c
    else:
        c = atoms.colors
        c[:, :3] = color.uint8x4()[:3]    # Preserve transparency
        if opacity is not None:
            c[:, 3] = opacity
        atoms.colors = c


def _set_ribbon_colors(residues, color, opacity, bgcolor=None):
    if color not in _SpecialColors:
        c = residues.ribbon_colors
        c[:, :3] = color.uint8x4()[:3]    # Preserve transparency
        if opacity is not None:
            c[:, 3] = opacity
        residues.ribbon_colors = c
    elif color == 'bychain':
        from ..atomic.colors import chain_colors
        c = chain_colors(residues.chain_ids)
        c[:, 3] = residues.ribbon_colors[:, 3] if opacity is None else opacity
        residues.ribbon_colors = c
    elif color == "bypolymer":
        from ..atomic.colors import polymer_colors
        c,rmask = polymer_colors(residues)
        c[rmask, 3] = residues.ribbon_colors[rmask, 3] if opacity is None else opacity
        residues.filter(rmask).ribbon_colors = c[rmask,:]
    elif color == 'bymodel':
        for m, res in residues.by_structure:
            c = res.ribbon_colors
            c[:, :3] = m.initial_color(bgcolor).uint8x4()[:3]
            if opacity is not None:
                c[:, 3] = opacity
            res.ribbon_colors = c
    elif color == 'random':
        from numpy import random, uint8
        c = random.randint(0, 255, (len(residues), 4)).astype(uint8)
        c[:, 3] = 255   # No transparency
        residues.ribbon_colors = c


def _set_surface_colors(session, atoms, color, opacity, bgcolor=None,
                        map=None, palette=None, range=None, offset=0):
    from .scolor import scolor
    if color in _SpecialColors:
        if color == 'fromatoms':
            ns = scolor(session, atoms, opacity=opacity, byatom=True)
        else:
            # Surface colored different from atoms
            c = _computed_atom_colors(atoms, color, opacity, bgcolor)
            ns = scolor(session, atoms, opacity=opacity, byatom=True, per_atom_colors=c)
            
    else:
        ns = scolor(session, atoms, color, opacity=opacity,
                    map=map, palette=palette, range=range, offset=offset)
    return ns

def _set_model_colors(session, m, color, map, opacity, palette, range, offset):
    if map is None:
        c = color.uint8x4()
        if not opacity is None:
            c[3] = opacity
        elif not m.single_color is None:
            c[3] = m.single_color[3]
        m.single_color = c
    else:
        if hasattr(m, 'surface_drawings_for_vertex_coloring'):
            surfs = m.surface_drawings_for_vertex_coloring()
        elif not m.empty_drawing():
            surfs = [m]
        else:
            surfs = []
        for s in surfs:
            from .scolor import volume_color_source
            cs = volume_color_source(s, map, palette, range, offset=offset)
            vcolors = cs.vertex_colors(s, session.logger.info)
            if opacity is not None:
                vcolors[:,3] = opacity
            else:
                vcolors[:,3] = s.color[3] if s.vertex_colors is None else s.vertex_colors[:,3]
            s.vertex_colors = vcolors

# -----------------------------------------------------------------------------
# Chain ids in each structure are colored from color map ordered alphabetically.
#
def _set_sequential_chain(session, selected, cmap, opacity, target):
    # Organize selected atoms by structure and then chain
    uc = selected.atoms.residues.chains.unique()
    chain_atoms = {}
    for c in uc:
        chain_atoms.setdefault(c.structure, []).append((c.chain_id, c.existing_residues.atoms))
    # Make sure there is a colormap
    if cmap is None:
        from .. import colors
        cmap = colors.BuiltinColormaps["rainbow"]
    # Each structure is colored separately with cmap applied by chain
    import numpy
    from ..colors import Color
    for sl in chain_atoms.values():
        sl.sort(key = lambda ca: ca[0])	# Sort by chain id
        colors = cmap.interpolated_rgba(numpy.linspace(0.0, 1.0, len(sl)))
        for color, (chain_id, atoms) in zip(colors, sl):
            c = Color(color)
            if target is None or 'a' in target:
                _set_atom_colors(atoms, c, opacity)
            if target is None or 'c' in target:
                res = atoms.unique_residues
                _set_ribbon_colors(res, c, opacity)
            if target is None or 's' in target:
                _set_surface_colors(session, atoms, c, opacity)

# ----------------------------------------------------------------------------------
# Polymers (unique sequences) in each structure are colored from color map ordered
# by polymer length.
#
def _set_sequential_polymer(session, objects, cmap, opacity, target):
    # Organize atoms by structure and then polymer sequence
    uc = objects.atoms.residues.chains.unique()
    seq_atoms = {}
    for c in uc:
        seq_atoms.setdefault(c.structure, {}).setdefault(c.characters, []).append(c.existing_residues.atoms)
    # Make sure there is a colormap
    if cmap is None:
        from .. import colors
        cmap = colors.BuiltinColormaps["rainbow"]
    # Each structure is colored separately with cmap applied by chain
    import numpy
    from ..colors import Color
    for sl in seq_atoms.values():
        sseq = list(sl.items())
        sseq.sort(key = lambda sa: len(sa[0]))	# Sort by sequence length
        colors = cmap.interpolated_rgba(numpy.linspace(0.0, 1.0, len(sseq)))
        for color, (seq, alist) in zip(colors, sseq):
            c = Color(color)
            for atoms in alist:
                if target is None or 'a' in target:
                    _set_atom_colors(atoms, c, opacity)
                if target is None or 'c' in target:
                    res = atoms.unique_residues
                    _set_ribbon_colors(res, c, opacity)
                if target is None or 's' in target:
                    _set_surface_colors(session, atoms, c, opacity)

# -----------------------------------------------------------------------------
#
def _set_sequential_residue(session, selected, cmap, opacity, target):
    # Make sure there is a colormap
    if cmap is None:
        from .. import colors
        cmap = colors.BuiltinColormaps["rainbow"]
    # Get chains and atoms in chains with "by_chain"
    # Each chain is colored separately with cmap applied by residue
    import numpy
    from ..colors import Color
    structure_chain_ids = {}
    for structure, chain_id, atoms in selected.atoms.by_chain:
        try:
            cids = structure_chain_ids[structure]
        except KeyError:
            structure_chain_ids[structure] = cids = set()
        cids.add(chain_id)
    for structure, cids in structure_chain_ids.items():
        for chain in structure.chains:
            if chain.chain_id not in cids:
                continue
            residues = chain.existing_residues
            colors = cmap.interpolated_rgba(numpy.linspace(0.0, 1.0, len(residues)))
            for color, r in zip(colors, residues):
                c = Color(color)
                if target is None or 'a' in target:
                    _set_atom_colors(r.atoms, c, opacity)
                if target is None or 'c' in target:
                    rgba = c.uint8x4()
                    if opacity is not None:
                        rgba[3] = opacity
                    r.ribbon_color = rgba

# -----------------------------------------------------------------------------
#
def _set_sequential_structures(session, selected, cmap, opacity, target):
    # Make sure there is a colormap
    if cmap is None:
        from .. import colors
        cmap = colors.BuiltinColormaps["rainbow"]

    from ..atomic import Structure
    models = list(m for m in selected.models if isinstance(m, Structure))
    models.sort(key = lambda m: m.id)
    if len(models) == 0:
        return

    # Each structure is colored separately with cmap applied by chain
    import numpy
    from ..colors import Color
    colors = cmap.interpolated_rgba(numpy.linspace(0.0, 1.0, len(models)))
    for color, m in zip(colors, models):
        c = Color(color)
        if 'a' in target:
            _set_atom_colors(m.atoms, c, opacity)
        if 'c' in target:
            _set_ribbon_colors(m.residues, c, opacity)
        if 's' in target:
            from .scolor import scolor
            ns = scolor(session, m.atoms, c)

# -----------------------------------------------------------------------------
#
_SequentialColor = {
    "polymers": _set_sequential_polymer,
    "chains": _set_sequential_chain,
    "residues": _set_sequential_residue,
    "structures": _set_sequential_structures,
}

# -----------------------------------------------------------------------------
#
def register_command(session):
    from . import register, CmdDesc, ColorArg, ColormapArg, ColormapRangeArg, ObjectsArg, create_alias
    from . import EmptyArg, Or, EnumOf, StringArg, TupleOf, FloatArg, BoolArg, AtomsArg
    from ..map import MapArg
    what_arg = EnumOf(('atoms', 'cartoons', 'ribbons', 'surfaces', 'bonds', 'pseudobonds'))
    desc = CmdDesc(required=[('objects', Or(ObjectsArg, EmptyArg))],
                   optional=[('color', Or(ColorArg, EnumOf(_SpecialColors))),
                             ('what', what_arg)],
                   keyword=[('target', StringArg),
                            ('transparency', FloatArg),
                            ('sequential', EnumOf(_SequentialLevels)),
                            ('halfbond', BoolArg),
                            ('map', MapArg),
                            ('palette', ColormapArg),
                            ('range', ColormapRangeArg),
                            ('offset', FloatArg),
                            ('zone', AtomsArg),
                            ('distance', FloatArg),
                   ],
                   synopsis="color objects")
    register('color', desc, color, logger=session.logger)
    create_alias('colour', 'color $*', logger=session.logger)
