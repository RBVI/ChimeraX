# vim: set expandtab shiftwidth=4 softtabstop=4:


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
    if structures is None:
        from chimerax.core.commands import atomspec
        results = atomspec.everything(session).evaluate(session)
        structures = results.atoms.unique_structures
    if scale is not None:
        structures.ribbon_tether_scales = scale
    if shape is not None:
        ts = _TetherShapeMap.get(shape, Structure.TETHER_CONE)
        structures.ribbon_tether_shapes = ts
    if sides is not None:
        structures.ribbon_tether_sides = sides
    if opacity is not None:
        structures.ribbon_tether_opacities = opacity


def cartoon_style(session, structures=None, helix=None, sheet=None, coil=None, nucleic=None):
    '''Set cartoon ribbon cross section style for specified structures.

    Parameters
    ----------
    structures : atomic structures
        Set options for selected atomic structures.
    helix : style
        Set helix cross section to style
    sheet : style
        Set sheet cross section to style
    coil : style
        Set coil cross section to style
    nucleic : style
        Set nucleic cross section to style
    '''
    if structures is None:
        from chimerax.core.commands import atomspec
        results = atomspec.everything(session).evaluate(session)
        structures = results.atoms.unique_structures
    for m in structures:
        if helix is not None:
            m.ribbon_xs_mgr.set_helix_style(helix)
        if sheet is not None:
            m.ribbon_xs_mgr.set_sheet_style(sheet)
        if coil is not None:
            m.ribbon_xs_mgr.set_coil_style(coil)
        if nucleic is not None:
            m.ribbon_xs_mgr.set_nucleic_style(nucleic)


def cartoon_scale(session, structures=None, helix=None, helix_arrow=None,
                  sheet=None, sheet_arrow=None, coil=None, nucleic=None):
    '''Set cartoon ribbon scale factors for specified structures.

    Parameters
    ----------
    structures : atomic structures
        Set options for selected atomic structures.
    helix : style
        Set helix cross section scale to 2-tuple of float
    helix_arrow : style
        Set helix arrow cross section scale to 4-tuple of float
    sheet : style
        Set sheet cross section scale to 2-tuple of float
    sheet_arrow : style
        Set sheet arrow cross section scale to 4-tuple of float
    coil : style
        Set coil cross section scale to 2-tuple of float
    nucleic : style
        Set nucleic cross section scale to 2-tuple of float
    '''
    if structures is None:
        from chimerax.core.commands import atomspec
        results = atomspec.everything(session).evaluate(session)
        structures = results.atoms.unique_structures
    for m in structures:
        if helix is not None:
            m.ribbon_xs_mgr.set_helix_scale(*helix)
        if helix_arrow is not None:
            m.ribbon_xs_mgr.set_helix_arrow_scale(*helix_arrow)
        if sheet is not None:
            m.ribbon_xs_mgr.set_sheet_scale(*sheet)
        if sheet_arrow is not None:
            m.ribbon_xs_mgr.set_sheet_arrow_scale(*sheet_arrow)
        if coil is not None:
            m.ribbon_xs_mgr.set_coil_scale(*coil)
        if nucleic is not None:
            m.ribbon_xs_mgr.set_nucleic_scale(*nucleic)


def cartoon_transition(session, structures, classes, ribbons):
    '''Set cartoon ribbon transitions for specified structures.

    Parameters
    ----------
    structures : atomic structures
        Set options for selected atomic structures.
    classes : 3-tuple of residue classes (strings)
        Set transition mode for middle residue of 3-residue sequence.
    ribbons : 2-tuple of ribbon types
        Front and back ribbon types for this transition.
    '''
    if structures is True:
        # Must be the NoArg case
        from chimerax.core.commands import atomspec
        results = atomspec.everything(session).evaluate(session)
        structures = results.atoms.unique_structures
    for m in structures:
        try:
            m.ribbon_xs_mgr.set_transition(*classes, *ribbons)
        except ValueError as e:
            from chimerax.core.errors import UserError
            raise UserError(str(e))


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

        from chimerax.core.atomic.ribbon import XSectionManager
        styles = EnumOf([XSectionManager.STYLE_SQUARE, XSectionManager.STYLE_ROUND],
                        ["square", "round"])
        desc = CmdDesc(optional=[("structures", AtomicStructuresArg)],
                       keyword=[("helix", styles),
                                ("sheet", styles),
                                ("coil", styles),
                                ("nucleic", styles),
                                ],
                       synopsis='set cartoon style options for specified structures')
        register(command_name + " style", desc, cartoon_style)

        desc = CmdDesc(optional=[("structures", AtomicStructuresArg)],
                       keyword=[("helix", TupleOf(FloatArg, 2)),
                                ("helix_arrow", TupleOf(FloatArg, 4)),
                                ("sheet", TupleOf(FloatArg, 2)),
                                ("sheet_arrow", TupleOf(FloatArg, 4)),
                                ("coil", TupleOf(FloatArg, 2)),
                                ("nucleic", TupleOf(FloatArg, 2)),
                                ],
                       synopsis='set cartoon scale options for specified structures')
        register(command_name + " scale", desc, cartoon_scale)

        classes = EnumOf([XSectionManager.RC_HELIX_START,
                          XSectionManager.RC_HELIX_START,
                          XSectionManager.RC_HELIX_MIDDLE,
                          XSectionManager.RC_HELIX_END,
                          XSectionManager.RC_HELIX_END,
                          XSectionManager.RC_SHEET_START,
                          XSectionManager.RC_SHEET_START,
                          XSectionManager.RC_SHEET_MIDDLE,
                          XSectionManager.RC_SHEET_END,
                          XSectionManager.RC_SHEET_END,
                          XSectionManager.RC_COIL],
                          ["start_helix", "sh", "helix", "end_helix", "eh",
                           "start_sheet", "ss", "sheet", "end_sheet", "es", "coil"])
        ribbons = EnumOf([XSectionManager.RIBBON_HELIX, XSectionManager.RIBBON_HELIX_ARROW,
                          XSectionManager.RIBBON_SHEET, XSectionManager.RIBBON_SHEET_ARROW,
                          XSectionManager.RIBBON_COIL],
                         ["helix", "arrow_helix", "sheet", "arrow_sheet", "coil"])
        desc = CmdDesc(required=[("structures", Or(AtomicStructuresArg, NoArg)),
                                 ("classes", TupleOf(classes, 3)),
                                 ("ribbons", TupleOf(ribbons, 2)),
                                 ],
                       synopsis='set cartoon transition options for specified structures')
        register(command_name + " transition", desc, cartoon_transition)
