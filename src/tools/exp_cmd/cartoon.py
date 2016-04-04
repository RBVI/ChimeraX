# vim: set expandtab shiftwidth=4 softtabstop=4:


from chimerax.core.atomic.ribbon import XSectionManager
from chimerax.core.atomic import Residue, Structure

_StyleMap = {
    "ribbon": Residue.RIBBON,
    "pipe": Residue.PIPE,
    "plank": Residue.PIPE,
    "pandp": Residue.PIPE,
}
_OrientMap = {
    "guides": Structure.RIBBON_ORIENT_GUIDES,
    "atoms": Structure.RIBBON_ORIENT_ATOMS,
    "curvature": Structure.RIBBON_ORIENT_CURVATURE,
}
_TetherShapeMap = {
    "cone": Structure.TETHER_CONE,
    "cylinder": Structure.TETHER_CYLINDER,
    "steeple": Structure.TETHER_REVERSE_CONE,
}


def cartoon(session, spec=None, smooth=None, style=None, hide_backbone=None, orient=None,
            show_spine=False):
    '''Display cartoon for specified residues.

    Parameters
    ----------
    spec : atom specifier
        Show ribbons for the specified residues. If no atom specifier is given then ribbons are shown
        for all residues.  Residues that are already shown as ribbons remain shown as ribbons.
    smooth : floating point number
        Adjustment factor for strand and helix smoothing.  A factor of zero means the
        cartoon will pass through the atom position.  A factor of one means the cartoon
        will pass through the "ideal" position, e.g., center of the cylinder that best
        fits a helix.  A factor of "default" means to return to default (0.7 for strands
        and 0 for everything else).
    style : string
        Set "Ribbon" style.  Value may be "ribbon" for normal ribbons, or one of "pipe",
        "plank", or "pandp" to display residues as pipes and planks.
    hide_backbone : boolean
        Set whether displaying a ribbon hides the sphere/ball/stick representation of
        backbone atoms.
    orient : string
        Choose which method to use for determining ribbon orientation FOR THE ENTIRE STRUCTURE.
        "guides" uses "guide" atoms like the carbonyl oxygens.
        "atoms" generates orientation from ribbon atoms like alpha carbons.
        "curvature" orients ribbon to be perpendicular to maximum curvature direction.
        "default" is to use "guides" if guide atoms are all present or "atoms" if not.
    show_spine : boolean
        Display ribbon "spine" (horizontal lines across center of ribbon).
        This parameter applies at the atomic structure level, so setting it for any residue
        sets it for the entire structure.
    '''
    if spec is None:
        from chimerax.core.commands import atomspec
        spec = atomspec.everything(session)
    results = spec.evaluate(session)
    residues = results.atoms.residues
    residues.ribbon_displays = True
    if smooth is not None:
        if smooth is "default":
            # Convert to C++ default value
            smooth = -1.0
        residues.ribbon_adjusts = smooth
    if style is not None:
        s = _StyleMap.get(style, Residue.RIBBON)
        residues.ribbon_styles = s
    if orient is not None:
        o = _OrientMap.get(orient, None)
        for m in residues.unique_structures:
            m.ribbon_orientation = o
    if hide_backbone is not None:
        residues.ribbon_hide_backbones = hide_backbone
    if show_spine is not None:
        residues.unique_structures.ribbon_show_spines = show_spine


def _get_structures(session, structures):
    if structures is None:
        from chimerax.core.commands import atomspec
        results = atomspec.everything(session).evaluate(session)
        structures = results.atoms.unique_structures
    return structures


def cartoon_tether(session, structures=None, scale=None, shape=None, sides=None, opacity=None):
    '''Set cartoon ribbon tether options for specified structures.

    Parameters
    ----------
    structures : atomic structures
        Set option for selected atomic structures.
    scale : floating point number
        Scale factor relative to atom display radius.  A scale factor of zero means the
        tether is not displayed.
        This parameter applies at the atomic structure level, so setting it for any residue
        sets it for the entire structure.
    shape : string
        Sets shape of tethers.  "cone" has point on ribbon and base at atom.
        "steeple" has point at atom and base on ribbon.  "cylinder" is bond-like.
        This parameter applies at the atomic structure level, so setting it for any residue
        sets it for the entire structure.
    sides : integer
        Number of sides for either the cylinder or cone base depending on tether shape.
        This parameter applies at the atomic structure level, so setting it for any residue
        sets it for the entire structure.
    opacity : floating point number
        Scale factor relative to atom opacity.
        This parameter applies at the atomic structure level, so setting it for any residue
        sets it for the entire structure.
    '''
    structures = _get_structures(session, structures)
    if scale is not None:
        structures.ribbon_tether_scales = scale
    if shape is not None:
        ts = _TetherShapeMap.get(shape, Structure.TETHER_CONE)
        structures.ribbon_tether_shapes = ts
    if sides is not None:
        structures.ribbon_tether_sides = sides
    if opacity is not None:
        structures.ribbon_tether_opacities = opacity


def cartoon_style(session, structures=None, helix=None, strand=None, coil=None, nucleic=None):
    '''Set cartoon ribbon cross section style for specified structures.

    Parameters
    ----------
    structures : atomic structures
        Set options for selected atomic structures.
    helix : style
        Set helix cross section to style
    strand : style
        Set strand cross section to style
    coil : style
        Set coil cross section to style
    nucleic : style
        Set nucleic cross section to style
    '''
    for m in _get_structures(session, structures):
        if helix is not None:
            m.ribbon_xs_mgr.set_helix_style(helix)
        if strand is not None:
            m.ribbon_xs_mgr.set_sheet_style(strand)
        if coil is not None:
            m.ribbon_xs_mgr.set_coil_style(coil)
        if nucleic is not None:
            m.ribbon_xs_mgr.set_nucleic_style(nucleic)


def cartoon_scale(session, structures=None, helix=None, arrow_helix=None,
                  strand=None, arrow_strand=None, coil=None, nucleic=None):
    '''Set cartoon ribbon scale factors for specified structures.

    Parameters
    ----------
    structures : atomic structures
        Set options for selected atomic structures.
    helix : style
        Set helix cross section scale to 2-tuple of float
    arrow_helix : style
        Set helix arrow cross section scale to 4-tuple of float
    strand : style
        Set strand cross section scale to 2-tuple of float
    arrow_strand : style
        Set strand arrow cross section scale to 4-tuple of float
    coil : style
        Set coil cross section scale to 2-tuple of float
    nucleic : style
        Set nucleic cross section scale to 2-tuple of float
    '''
    for m in _get_structures(session, structures):
        if helix is not None:
            m.ribbon_xs_mgr.set_helix_scale(*helix)
        if arrow_helix is not None:
            m.ribbon_xs_mgr.set_helix_arrow_scale(*arrow_helix)
        if strand is not None:
            m.ribbon_xs_mgr.set_sheet_scale(*strand)
        if arrow_strand is not None:
            m.ribbon_xs_mgr.set_sheet_arrow_scale(*arrow_strand)
        if coil is not None:
            m.ribbon_xs_mgr.set_coil_scale(*coil)
        if nucleic is not None:
            m.ribbon_xs_mgr.set_nucleic_scale(*nucleic)


def cartoon_linker(session, structures, classes, linker):
    '''Set cartoon ribbon transitions for specified structures.

    Parameters
    ----------
    structures : atomic structures
        Set options for selected atomic structures.
    classes : 2-tuple of residue classes (strings)
        Set linker mode from one type of secondary structure to another
    linker : string
        Type of linker to use between secondary structures
    '''
    RC_HELIX_START = XSectionManager.RC_HELIX_START
    RC_HELIX_MIDDLE = XSectionManager.RC_HELIX_MIDDLE
    RC_HELIX_END = XSectionManager.RC_HELIX_END
    RC_SHEET_START = XSectionManager.RC_SHEET_START
    RC_SHEET_MIDDLE = XSectionManager.RC_SHEET_MIDDLE
    RC_SHEET_END = XSectionManager.RC_SHEET_END
    RC_COIL = XSectionManager.RC_COIL
    RIBBON_HELIX = XSectionManager.RIBBON_HELIX
    RIBBON_HELIX_ARROW = XSectionManager.RIBBON_HELIX_ARROW
    RIBBON_SHEET = XSectionManager.RIBBON_SHEET
    RIBBON_SHEET_ARROW = XSectionManager.RIBBON_SHEET_ARROW
    RIBBON_COIL = XSectionManager.RIBBON_COIL
    from chimerax.core.errors import UserError
    if classes[0] == "helix":
        if classes[1] == "helix":
            rc_list = [[(RC_HELIX_MIDDLE, RC_HELIX_END, RC_HELIX_START),
                        (RC_HELIX_START, RC_HELIX_END, RC_HELIX_START)],
                       [(RC_HELIX_END, RC_HELIX_START, RC_HELIX_MIDDLE),
                        (RC_HELIX_END, RC_HELIX_START, RC_HELIX_END)]]
        elif classes[1] == "strand":
            rc_list = [[(RC_HELIX_MIDDLE, RC_HELIX_END, RC_SHEET_START),
                        (RC_HELIX_START, RC_HELIX_END, RC_SHEET_START)],
                       [(RC_HELIX_END, RC_SHEET_START, RC_SHEET_MIDDLE),
                        (RC_HELIX_END, RC_SHEET_START, RC_SHEET_END)]]
        elif classes[1] == "coil":
            rc_list = [[(RC_HELIX_MIDDLE, RC_HELIX_END, RC_COIL),
                        (RC_HELIX_START, RC_HELIX_END, RC_COIL)],
                       []]
        else:
            raise UserError("unsupported linker %s-%s" % classes)
    elif classes[0] == "strand":
        if classes[1] == "helix":
            rc_list = [[(RC_SHEET_MIDDLE, RC_SHEET_END, RC_HELIX_START),
                        (RC_SHEET_START, RC_SHEET_END, RC_HELIX_START)],
                       [(RC_SHEET_END, RC_HELIX_START, RC_HELIX_MIDDLE),
                        (RC_SHEET_END, RC_HELIX_START, RC_HELIX_END)]]
        elif classes[1] == "strand":
            rc_list = [[(RC_SHEET_MIDDLE, RC_SHEET_END, RC_SHEET_START),
                        (RC_SHEET_START, RC_SHEET_END, RC_SHEET_START)],
                       [(RC_SHEET_END, RC_SHEET_START, RC_SHEET_MIDDLE),
                        (RC_SHEET_END, RC_SHEET_START, RC_SHEET_END)]]
        elif classes[1] == "coil":
            rc_list = [[(RC_SHEET_MIDDLE, RC_SHEET_END, RC_COIL),
                        (RC_SHEET_START, RC_SHEET_END, RC_COIL)],
                       []]
        else:
            raise UserError("unsupported linker %s-%s" % classes)
    elif classes[0] == "coil":
        if classes[1] == "helix":
            rc_list = [[],
                       [(RC_COIL, RC_HELIX_START, RC_HELIX_MIDDLE),
                        (RC_COIL, RC_HELIX_START, RC_HELIX_END)]]
        elif classes[1] == "strand":
            rc_list = [[],
                       [(RC_COIL, RC_SHEET_START, RC_SHEET_MIDDLE),
                        (RC_COIL, RC_SHEET_START, RC_SHEET_END)]]
        else:
            raise UserError("unsupported linker %s-%s" % classes)
    if linker == "long":
        if classes[0] == "helix":
            transition_list = [(RIBBON_HELIX_ARROW, RIBBON_COIL)]
        elif classes[0] == "strand":
            transition_list = [(RIBBON_SHEET_ARROW, RIBBON_COIL)]
        else:
            transition_list = [None]
        if classes[1] == "helix":
            transition_list.append((RIBBON_COIL, RIBBON_HELIX))
        elif classes[1] == "strand":
            transition_list.append((RIBBON_COIL, RIBBON_SHEET))
        else:
            transition_list.append(None)
    elif linker == "short":
        if classes[0] == "helix":
            transition_list = [(RIBBON_HELIX, RIBBON_HELIX_ARROW)]
        elif classes[0] == "strand":
            transition_list = [(RIBBON_SHEET, RIBBON_SHEET_ARROW)]
        else:
            transition_list = [None]
        if classes[1] == "helix":
            transition_list.append((RIBBON_COIL, RIBBON_HELIX))
        elif classes[1] == "strand":
            transition_list.append((RIBBON_COIL, RIBBON_SHEET))
        else:
            transition_list.append(None)
    elif linker == "short2":
        if classes[0] == "helix":
            transition_list = [(RIBBON_HELIX_ARROW, RIBBON_COIL)]
        elif classes[0] == "strand":
            transition_list = [(RIBBON_SHEET_ARROW, RIBBON_COIL)]
        else:
            transition_list = [None]
        if classes[1] == "helix":
            transition_list.append((RIBBON_HELIX, RIBBON_HELIX))
        elif classes[1] == "strand":
            transition_list.append((RIBBON_SHEET, RIBBON_SHEET))
        else:
            transition_list.append(None)
    elif linker == "none":
        if classes[0] == "helix":
            transition_list = [(RIBBON_HELIX, RIBBON_HELIX_ARROW)]
        elif classes[0] == "strand":
            transition_list = [(RIBBON_SHEET, RIBBON_SHEET_ARROW)]
        else:
            transition_list = [None]
        if classes[1] == "helix":
            transition_list.append((RIBBON_HELIX, RIBBON_HELIX))
        elif classes[1] == "strand":
            transition_list.append((RIBBON_SHEET, RIBBON_SHEET))
        else:
            transition_list.append(None)
    else:
        raise UserError("unknown linker %s" % linker)
    for m in _get_structures(session, structures):
        try:
            for rcs, links in zip(rc_list, transition_list):
                if links is not None:
                    for rc in rcs:
                        m.ribbon_xs_mgr.set_transition(*rc, *links)
        except ValueError as e:
            raise UserError(str(e))


def cartoon_arrow(session, structures=None, helix=None, strand=None):
    '''Set cartoon ribbon arrow display for specified structures.

    Parameters
    ----------
    structures : atomic structures
        Set options for selected atomic structures.
    helix : boolean
        Set helix cross section scale to 2-tuple of float
    strand : boolean
        Set strand cross section scale to 2-tuple of float
    '''
    for m in _get_structures(session, structures):
        if helix is not None:
            m.ribbon_xs_mgr.set_helix_end_arrow(helix)
        if strand is not None:
            m.ribbon_xs_mgr.set_sheet_end_arrow(strand)


def cartoon_param_round(session, structures=None, faceted=None, sides=None):
    '''Set cartoon round ribbon parameters for specified structures.

    Parameters
    ----------
    structures : atomic structures
        Set options for selected atomic structures.
    faceted : boolean
        Set whether highlights are per-vertex or per-face.
    sides : integer
        Set number of sides in cross section.
    '''
    for m in _get_structures(session, structures):
        if faceted is not None:
            m.ribbon_xs_mgr.set_round_param("faceted", faceted)
        if sides is not None:
            m.ribbon_xs_mgr.set_round_param("sides", sides)


def cartoon_param_piped(session, structures=None, faceted=None, sides=None, ratio=None):
    '''Set cartoon piped ribbon parameters for specified structures.

    Parameters
    ----------
    structures : atomic structures
        Set options for selected atomic structures.
    faceted : boolean
        Set whether highlights are per-vertex or per-face.
    sides : integer
        Set number of sides in cross section.
    ratio : real number
        Set thickness ratio between flat center and piping.
    '''
    for m in _get_structures(session, structures):
        if faceted is not None:
            m.ribbon_xs_mgr.set_piped_param("faceted", faceted)
        if sides is not None:
            m.ribbon_xs_mgr.set_piped_param("sides", sides)
        if ratio is not None:
            m.ribbon_xs_mgr.set_piped_param("ratio", ratio)


def uncartoon(session, spec=None):
    '''Undisplay ribbons for specified residues.

    Parameters
    ----------
    spec : atom specifier
        Hide ribbons for the specified residues. If no atom specifier is given then all ribbons are hidden.
    '''
    if spec is None:
        from chimerax.core.commands import atomspec
        spec = atomspec.everything(session)
    results = spec.evaluate(session)
    results.atoms.residues.ribbon_displays = False


def initialize(command_name):
    from chimerax.core.commands import register
    from chimerax.core.commands import CmdDesc, AtomSpecArg, AtomicStructuresArg
    if command_name.startswith('~'):
        desc = CmdDesc(optional=[("spec", AtomSpecArg)],
                       synopsis='undisplay cartoon for specified residues')
        register(command_name, desc, uncartoon)
    else:
        from chimerax.core.commands import Or, Bounded, FloatArg, EnumOf, BoolArg, IntArg, TupleOf, NoArg
        desc = CmdDesc(optional=[("spec", AtomSpecArg)],
                       keyword=[("smooth", Or(Bounded(FloatArg, 0.0, 1.0),
                                              EnumOf(["default"]))),
                                ("style", EnumOf(list(_StyleMap.keys()))),
                                ("orient", EnumOf(list(_OrientMap.keys()))),
                                ("hide_backbone", BoolArg),
                                ("show_spine", BoolArg),
                                ],
                       synopsis='display cartoon for specified residues')
        register(command_name, desc, cartoon)

        desc = CmdDesc(optional=[("structures", AtomicStructuresArg)],
                       keyword=[("scale", Bounded(FloatArg, 0.0, 1.0)),
                                ("shape", EnumOf(_TetherShapeMap.keys())),
                                ("sides", Bounded(IntArg, 3, 10)),
                                ("opacity", Bounded(FloatArg, 0.0, 1.0)),
                                ],
                       synopsis='set cartoon tether options for specified structures')
        register(command_name + " tether", desc, cartoon_tether)

        styles = EnumOf([XSectionManager.STYLE_SQUARE, XSectionManager.STYLE_ROUND],
                        ["square", "round"])
        desc = CmdDesc(optional=[("structures", AtomicStructuresArg)],
                       keyword=[("helix", styles),
                                ("strand", styles),
                                ("coil", styles),
                                ("nucleic", styles),
                                ],
                       synopsis='set cartoon style options for specified structures')
        register(command_name + " style", desc, cartoon_style)

        desc = CmdDesc(optional=[("structures", AtomicStructuresArg)],
                       keyword=[("helix", TupleOf(FloatArg, 2)),
                                ("arrow_helix", TupleOf(FloatArg, 4)),
                                ("strand", TupleOf(FloatArg, 2)),
                                ("arrow_strand", TupleOf(FloatArg, 4)),
                                ("coil", TupleOf(FloatArg, 2)),
                                ("nucleic", TupleOf(FloatArg, 2)),
                                ],
                       synopsis='set cartoon scale options for specified structures')
        register(command_name + " scale", desc, cartoon_scale)

        classes = EnumOf(["helix", "strand", "coil"])
        linker = EnumOf(["none", "short", "short2", "long"])
        desc = CmdDesc(required=[("structures", Or(AtomicStructuresArg, NoArg)),
                                 ("linker", linker),
                                 ("classes", TupleOf(classes, 2)),
                                 ],
                       synopsis='set cartoon linker options for specified structures')
        register(command_name + " linker", desc, cartoon_linker)

        desc = CmdDesc(optional=[("structures", AtomicStructuresArg)],
                       keyword=[("helix", BoolArg),
                                ("strand", BoolArg),
                                ],
                       synopsis='set cartoon arrow options for specified structures')
        register(command_name + " arrow", desc, cartoon_arrow)

        desc = CmdDesc(optional=[("structures", AtomicStructuresArg)],
                       keyword=[("sides", Bounded(IntArg, 3)),
                                ("faceted", BoolArg),
                                ],
                       synopsis='set cartoon round ribbon options for specified structures')
        register(command_name + " param round", desc, cartoon_param_round)

        desc = CmdDesc(optional=[("structures", AtomicStructuresArg)],
                       keyword=[("sides", Bounded(IntArg, 3)),
                                ("faceted", BoolArg),
                                ("ratio", Bounded(FloatArg, 0.2, 0.8)),
                                ],
                       synopsis='set cartoon piped ribbon options for specified structures')
        register(command_name + " param piped", desc, cartoon_param_piped)
