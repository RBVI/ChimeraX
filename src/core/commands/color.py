# vi: set expandtab shiftwidth=4 softtabstop=4:

_SpecialColors = ["byatom", "byelement", "byhetero", "bychain"]

_SequentialLevels = ["residues", "helix", "helices", "strands",
                     "SSEs", "chains", "molmodels",
                     "volmodels", "allmodels"]

_CmapRanges = ["full"]

def color(session, spec, color=None, target=None,
           sequential=None, cmap=None, cmap_range=None):
    """Color atoms, ribbons, surfaces, ....

    Parameters
    ----------
    spec : specifier
      Which objects to color.
    color : Color
      Color can be a standard color name or "byelement", "byhetero" or "bychain" .
    target : string
      Characters indicating what to color, a = atoms, c = cartoon, s = surfaces, m = models,
      n = non-molecule models, l = labels, r = residue labels, b = bonds, p = pseudobonds, d = distances.
      Everything is colored if no target is specified.
    sequential : string
      Value can only be "chains", assigns each chain a color from a color map.
    cmap : Colormap
      Color map to use with sequential coloring
    cmap_range : 2 comma-separated floats or "full"
      Specifies the range of value used for sampling from a color map.
    """
    from . import atomspec
    if spec is None:
        spec = atomspec.everything(session)
    results = spec.evaluate(session)
    if sequential is not None:
        try:
            f = _SequentialColor[sequential]
        except KeyError:
            from ..errors import UserError
            raise UserError("sequential \"%s\" not implemented yet"
                            % sequential)
        else:
            f(results, cmap, target)
            return
    what = []

    if target is None or 'a' in target:
        # atoms/bonds
        atoms = results.atoms
        if atoms is not None:
            if color in _SpecialColors:
                if color == "byelement":
                    _set_element_colors(atoms, False)
                elif color == "byhetero":
                    _set_element_colors(atoms, True)
                elif color == "bychain":
                    from .. import colors
                    colors.color_atoms_by_chain(atoms)
                else:
                    # Other "colors" do not apply to atoms
                    pass
            else:
                atoms.colors = color.uint8x4()
            what.append('%d atoms' % len(atoms))

    if target is None or 'l' in target:
        if target is not None:
            session.logger.warning('Label colors not supported yet')

    if target is None or 's' in target:
        from .scolor import scolor
        if color in _SpecialColors:
            ns = scolor(session, results.atoms, byatom=True)
        else:
            ns = scolor(session, results.atoms, color)
        what.append('%d surfaces' % ns)

    if target is None or 'c' in target:
        residues = results.atoms.unique_residues
        if color not in _SpecialColors:
            residues.ribbon_colors = color.uint8x4()
        elif color == 'bychain':
            from .. import colors
            colors.color_ribbons_by_chain(residues)
        what.append('%d residues' % len(residues))

    if target is None or 'r' in target:
        if target is not None:
            session.logger.warning('Residue label colors not supported yet')

    if target is None or 'n' in target:
        if target is not None:
            session.logger.warning('Non-molecular model-level colors not supported yet')

    if target is None or 'm' in target:
        if target is not None:
            session.logger.warning('Model-level colors not supported yet')

    if target is None or 'b' in target:
        if target is not None:
            session.logger.warning('Bond colors not supported yet')

    if target is None or 'p' in target:
        if target is not None:
            session.logger.warning('Pseudobond colors not supported yet')

    if target is None or 'd' in target:
        if target is not None:
            session.logger.warning('Distances colors not supported yet')

    if not what:
        what.append('nothing')

    from . import cli
    session.logger.status('Colored %s' % cli.commas(what, ' and')[0])


def _set_element_colors(atoms, skip_carbon):
    import numpy
    en = atoms.element_numbers
    for e in numpy.unique(en):
        if not skip_carbon or e != 6:
            from .. import colors
            ae = atoms.filter(en == e)
            ae.colors = colors.element_colors(e)


# -----------------------------------------------------------------------------
#
def _set_sequential_chain(selected, cmap, target):
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
            c = Color(color).uint8x4()
            if target is None or 'a' in target:
                atoms.colors = c
            if target is None or 'c' in target:
                atoms.unique_residues.ribbon_colors = c

_SequentialColor = {
    "chains": _set_sequential_chain,
}

# -----------------------------------------------------------------------------
#
def register_command(session):
    from . import register, CmdDesc, ColorArg, ColormapArg, AtomSpecArg
    from . import EmptyArg, Or, EnumOf, StringArg, TupleOf, FloatArg
    desc = CmdDesc(required=[('spec', Or(AtomSpecArg, EmptyArg))],
                   optional=[('color', Or(ColorArg, EnumOf(_SpecialColors)))],
                   keyword=[('target', StringArg),
                            ('sequential', EnumOf(_SequentialLevels)),
                            ('cmap', ColormapArg),
                            ('cmap_range', Or(TupleOf(FloatArg, 2), EnumOf(_CmapRanges)))],
                   synopsis="color objects")
    register('color', desc, color)
