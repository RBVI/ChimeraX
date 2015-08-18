# vi: set expandtab shiftwidth=4 softtabstop=4:

from . import cli
from ..colors import Color
from .colorarg import ColorArg, ColormapArg

_SpecialColors = ["byatom", "byelement", "byhetero", "bychain"]

_SequentialLevels = ["residues", "helix", "helices", "strands",
                     "SSEs", "chains", "molmodels", 
                     "volmodels", "allmodels"]

_CmapRanges = ["full"]

def define_color(session, name, color=None):
    """Create a user defined color."""
    if ' ' in name:
        from ..errors import UsetError
        raise UserError("Sorry, spaces are not alllowed in color names")
    if color is None:
        if session is not None:
            i = session.user_colors.bisect_left(name)
            if i < len(session.user_colors):
                real_name = session.user_colors.iloc[i]
                if real_name.startswith(name):
                    color = session.user_colors[real_name]
        if color is None:
            from ..colors import BuiltinColors
            i = BuiltinColors.bisect_left(name)
            if i < len(BuiltinColors):
                real_name = BuiltinColors.iloc[i]
                if real_name.startswith(name):
                    color = Color([x / 255 for x in BuiltinColors[real_name]])
        if color is None:
            session.logger.status('Unknown color %r' % name)
            return

        def percent(x):
            if x == 1:
                return 100
            return ((x * 10000) % 10000) / 100
        red, green, blue, alpha = color.rgba
        if alpha >= 1:
            transmit = 'opaque'
        elif alpha <= 0:
            transmit = 'transparent'
        else:
            transmit = '%g%% transparent' % percent(1 - alpha)

        msg = 'Color %r is %s, %.4g%% red, %.4g%% green, and %.4g%% blue' % (
            real_name, transmit, percent(red), percent(green),
            percent(blue))
        if session is None:
            print(msg)
            return
        session.logger.status(msg)
        session.logger.info(
            msg +
            '<div style="width:1em; height:.4em;'
            ' display:inline-block;'
            ' border:1px solid #000; background-color:%s"/>'
            % color.hex())
        return
    session.user_colors[name] = color


def undefine_color(session, name):
    """Remove a user defined color."""
    del session.user_colors[name]


def color(session, color, spec=None):
    """Color an object specification."""
    from . import atomspec
    if spec is None:
        spec = atomspec.everything(session)
    results = spec.evaluate(session)

    rgba8 = color.uint8x4()
    atoms = results.atoms
    if atoms is None:
        na = 0
    else:
        atoms.colors = rgba8
        na = len(atoms)

    ns = 0
    from ..structure import AtomicStructure
    for m in results.models:
        if not isinstance(m, AtomicStructure):
            m.color = rgba8
            ns += 1

    what = []
    if na > 0:
        what.append('%d atoms' % na)
    if ns > 0:
        what.append('%d surfaces' % ns)
    if na == 0 and ns == 0:
        what.append('nothing')
    session.logger.status('Colored %s' % ', '.join(what))


def rcolor(session, color, spec=None):
    """Color ribbons for an object specification."""
    from . import atomspec
    if spec is None:
        spec = atomspec.everything(session)
    results = spec.evaluate(session)

    rgba8 = color.uint8x4()
    residues = results.atoms.unique_residues
    if residues is None:
        nr = 0
    else:
        residues.ribbon_colors = rgba8
        nr = len(residues)

    what = []
    if nr > 0:
        what.append('%d residues' % nr)
    else:
        what.append('nothing')
    session.logger.status('Colored %s' % ', '.join(what))


def ecolor(session, spec, color=None, target=None,
           sequential=None, cmap=None, cmap_range=None):
    """Color an object specification."""
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
    session.logger.status('Colored %s' % ', '.join(what))


def _set_element_colors(atoms, skip_carbon):
    import numpy
    en = atoms.element_numbers
    for e in numpy.unique(en):
        if not skip_carbon or e != 6:
            ae = atoms.filter(en == e)
            from .. import colors
            atoms.filter(en == e).colors = colors.element_colors(e)

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
    from . import atomspec
    cli.register(
        'color',
        cli.CmdDesc(required=[("color", ColorArg)],
                    optional=[("spec", atomspec.AtomSpecArg)],
                    synopsis="color specified objects"),
        color
    )
    cli.register(
        'rcolor',
        cli.CmdDesc(required=[("color", ColorArg)],
                    optional=[("spec", atomspec.AtomSpecArg)],
                    synopsis="color specified ribbons"),
        rcolor
    )
    cli.register(
        'colordef',
        cli.CmdDesc(required=[('name', cli.StringArg)],
                    optional=[('color', ColorArg)],
                    synopsis="define a custom color"),
        define_color
    )
    cli.register(
        '~colordef',
        cli.CmdDesc(required=[('name', cli.StringArg)],
                    synopsis="remove color definition"),
        undefine_color
    )
    cli.register(
        'ecolor',
        cli.CmdDesc(required=[('spec', cli.Or(atomspec.AtomSpecArg, cli.EmptyArg))],
                    optional=[('color', cli.Or(ColorArg, cli.EnumOf(_SpecialColors)))],
                    keyword=[('target', cli.StringArg),
                             ('sequential', cli.EnumOf(_SequentialLevels)),
                             ('cmap', ColormapArg),
                             ('cmap_range', cli.Or(cli.TupleOf(cli.FloatArg, 2),
                                                    cli.EnumOf(_CmapRanges)))],
                    synopsis="testing real color syntax"),
        ecolor
    )


def test():
    tests = [
        "0x00ff00",
        "#0f0",
        "#00ffff",
        "gray(50)",
        "gray(50%)",
        "rgb(0, 0, 255)",
        "rgb(100%, 0, 0)",
        "red",
        "hsl(0, 100%, 50%)",  # red
        "lime",
        "hsl(120deg, 100%, 50%)",  # lime
        "darkgreen",
        "hsl(120, 100%, 20%)",  # darkgreen
        "lightgreen",
        "hsl(120, 75%, 75%)",  # lightgreen
    ]
    for t in tests:
        print(t)
        try:
            print(ColorArg.parse(t))
        except ValueError as err:
            print(err)
    print('same:', ColorArg.parse('white')[0] == Color('#ffffff'))
