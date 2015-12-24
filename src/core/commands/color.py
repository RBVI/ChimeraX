# vim: set expandtab shiftwidth=4 softtabstop=4:

_SpecialColors = ["byatom", "byelement", "byhetero", "bychain", "bymodel",
                  "fromatoms", "random"]

_SequentialLevels = ["residues", "helix", "helices", "strands",
                     "SSEs", "chains", "molmodels",
                     "volmodels", "allmodels"]

_CmapRanges = ["full"]


def color(session, objects, color=None, target=None, transparency=None,
          sequential=None, cmap=None, cmap_range=None, halfbond=None):
    """Color atoms, ribbons, surfaces, ....

    Parameters
    ----------
    objects : Objects
      Which objects to color.
    color : Color
      Color can be a standard color name or "byatom", "byelement", "byhetero", "bychain", "bymodel".
    target : string
      Characters indicating what to color, a = atoms, c = cartoon, s = surfaces, m = models,
      n = non-molecule models, l = labels, r = residue labels, b = bonds, p = pseudobonds, d = distances.
      Everything is colored if no target is specified.
    transparency : float
      Percent transparency to use.  If not specified current transparency is preserved.
    sequential : string
      Value can only be "chains", assigns each chain a color from a color map.
    cmap : :class:`.Colormap`
      Color map to use with sequential coloring.
    cmap_range : 2 comma-separated floats or "full"
      Specifies the range of value used for sampling from a color map.
    halfbond : bool
      Whether to color each half of a bond to match the connected atoms.
      If halfbond is false the bond is given the single color assigned to the bond.
    """
    if objects is None:
        from . import all_objects
        objects = all_objects(session)
    atoms = objects.atoms
    if color == "byhetero":
        atoms = atoms.filter(atoms.element_numbers != 6)

    default_target = (target is None)
    if default_target:
        target = 'acsmnlrbd'

    # Decide whether to set or preserve transparency
    opacity = None
    if transparency is not None:
        opacity = min(255, max(0, int(2.56 * (100 - transparency))))
    if getattr(color, 'explicit_transparency', False):
        opacity = color.uint8x4()[3]

    if sequential is not None:
        try:
            f = _SequentialColor[sequential]
        except KeyError:
            from ..errors import UserError
            raise UserError("sequential \"%s\" not implemented yet"
                            % sequential)
        else:
            f(objects, cmap, opacity, target)
            return

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

    if 's' in target and color is not None:
        from ..atomic import MolecularSurface, concatenate
        msatoms = [m.atoms for m in objects.models
                   if isinstance(m, MolecularSurface) and not m.atoms.intersects(atoms)]
        satoms = concatenate(msatoms + [atoms]) if msatoms else atoms
        if color == "byhetero":
            satoms = satoms.filter(satoms.element_numbers != 6)
        ns = _set_surface_colors(session, satoms, color, opacity, bgcolor)
        what.append('%d surfaces' % ns)

    if 'c' in target and color is not None:
        residues = atoms.unique_residues
        _set_ribbon_colors(residues, color, opacity, bgcolor)
        what.append('%d residues' % len(residues))

    if 'r' in target:
        if not default_target:
            session.logger.warning('Residue label colors not supported yet')

    if 'n' in target:
        if not default_target:
            session.logger.warning('Non-molecular model-level colors not supported yet')

    if 'm' in target:
        if not default_target:
            session.logger.warning('Model-level colors not supported yet')

    if 'b' in target:
        if atoms is not None:
            bonds = atoms.inter_bonds
            if len(bonds) > 0:
                if color not in _SpecialColors and color is not None:
                    bonds.colors = color.uint8x4()
                if halfbond is not None:
                    bonds.halfbonds = halfbond
                what.append('%d bonds' % len(bonds))

    if halfbond is not None and 'b' not in target and 'p' not in target and atoms is not None:
        bonds = atoms.inter_bonds
        if len(bonds) > 0:
            bonds.halfbonds = halfbond
            what.append('%d halfbonds' % len(bonds))

    if 'p' in target:
        if atoms is not None:
            from .. import atomic
            bonds = atomic.interatom_pseudobonds(atoms, session)
            if len(bonds) > 0:
                if color not in _SpecialColors and color is not None:
                    bonds.colors = color.uint8x4()
                if halfbond is not None:
                    bonds.halfbonds = halfbond
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
        from ..colors import chain_colors
        c = chain_colors(atoms.residues.chain_ids)
        c[:, 3] = atoms.colors[:, 3] if opacity is None else opacity
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
    from ..colors import element_colors
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
        from ..colors import chain_colors
        c = chain_colors(residues.chain_ids)
        c[:, 3] = residues.ribbon_colors[:, 3] if opacity is None else opacity
        residues.ribbon_colors = c
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


def _set_surface_colors(session, atoms, color, opacity, bgcolor):
    from .scolor import scolor
    if color in _SpecialColors:
        if color == 'fromatoms':
            ns = scolor(session, atoms, opacity=opacity, byatom=True)
        else:
            # Surface colored different from atoms
            c = _computed_atom_colors(atoms, color, opacity, bgcolor)
            ns = scolor(session, atoms, opacity=opacity, byatom=True, per_atom_colors=c)
    else:
        ns = scolor(session, atoms, color, opacity=opacity)
    return ns


# -----------------------------------------------------------------------------
#
def _set_sequential_chain(selected, cmap, opacity, target):
    # Organize selected atoms by structure and then chain
    sa = selected.atoms
    chain_atoms = sa.filter(sa.in_chains)
    structures = {}
    for structure, chain_id, atoms in chain_atoms.by_chain:
        try:
            sl = structures[structure]
        except KeyError:
            sl = []
            structures[structure] = sl
        sl.append((chain_id, atoms))
    # Make sure there is a colormap
    if cmap is None:
        from .. import colors
        cmap = colors.BuiltinColormaps["rainbow"]
    # Each structure is colored separately with cmap applied by chain
    import numpy
    from ..colors import Color
    for sl in structures.values():
        colors = cmap.get_colors_for(numpy.linspace(0.0, 1.0, len(sl)))
        for color, (chain_id, atoms) in zip(colors, sl):
            c = Color(color)
            if target is None or 'a' in target:
                _set_atom_colors(atoms, c, opacity)
            if target is None or 'c' in target:
                res = atoms.unique_residues
                _set_ribbon_colors(res, c, opacity)

_SequentialColor = {
    "chains": _set_sequential_chain,
}


# -----------------------------------------------------------------------------
#
def register_command(session):
    from . import register, CmdDesc, ColorArg, ColormapArg, ObjectsArg
    from . import EmptyArg, Or, EnumOf, StringArg, TupleOf, FloatArg, BoolArg
    desc = CmdDesc(required=[('objects', Or(ObjectsArg, EmptyArg))],
                   optional=[('color', Or(ColorArg, EnumOf(_SpecialColors)))],
                   keyword=[('target', StringArg),
                            ('transparency', FloatArg),
                            ('sequential', EnumOf(_SequentialLevels)),
                            ('cmap', ColormapArg),
                            ('cmap_range', Or(TupleOf(FloatArg, 2), EnumOf(_CmapRanges))),
                            ('halfbond', BoolArg)],
                   synopsis="color objects")
    register('color', desc, color)
