# vim: set expandtab shiftwidth=4 softtabstop=4:


from chimerax.core.atomic.ribbon import XSectionManager
from chimerax.core.atomic import Residue, Structure
from chimerax.core.commands import Annotation, AnnotationError

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
    "peptide": Structure.RIBBON_ORIENT_PEPTIDE,
}
_TetherShapeMap = {
    "cone": Structure.TETHER_CONE,
    "cylinder": Structure.TETHER_CYLINDER,
    "steeple": Structure.TETHER_REVERSE_CONE,
}
_XSectionMap = {
    "rectangle": XSectionManager.STYLE_SQUARE,
    "oval": XSectionManager.STYLE_ROUND,
    "barbell": XSectionManager.STYLE_PIPING,
    # Old names (to be removed)
    "square": XSectionManager.STYLE_SQUARE,
    "round": XSectionManager.STYLE_ROUND,
    "piping": XSectionManager.STYLE_PIPING,
}
_XSInverseMap = dict([(v, k) for k, v in _XSectionMap.items()])


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
        "peptide" orients ribbon to be perpendicular to peptide planes.
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
    residues.atoms.update_ribbon_visibility()


def _get_structures(session, structures):
    if structures is None or structures is True:
        # True is the NoArg case
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


def cartoon_style(session, spec=None, width=None, thickness=None, arrows=None, arrows_helix=None,
                  arrow_scale=None, xsection=None, sides=None,
                  bar_scale=None, bar_sides=None, ss_ends=None):
    '''Set cartoon style options for secondary structures in specified structures.

    Parameters
    ----------
    spec : atom specifier
        Set style for all secondary structure types that include the specified residues.
        If no atom specifier is given then style is set for all secondary structure types.
    width : floating point number
        Width of ribbons in angstroms.
    thickness : floating point number
        Thickness of ribbons in angstroms.
    arrows : boolean
        Whether to show arrow at ends of strands.
    arrows_helix : boolean
        Whether to show arrow at ends of helices.
    arrow_scale : floating point number
        Scale factor of arrow base width relative to strand or helix width.
    xsection : string
        Cross section type, one of "rectangle", "oval" or "barbell".
    sides : integer
        Number of sides for oval cross sections.
    bar_scale : floating point number
        Scale factor of barbell connector to ends.
    bar_sides : integer
        Number of sides for barbell cross sections.
    ss_ends : string
        Length of helix/strand representation relative to backbone atoms.
        One of "default", "short" or "long".
    '''
    if spec is None:
        from chimerax.core.commands import atomspec
        spec = atomspec.everything(session)
    results = spec.evaluate(session)
    structures = results.atoms.unique_structures
    if (width is None and thickness is None and arrows is None and
        arrows_helix is None and arrow_scale is None and xsection is None and
        sides is None and bar_scale is None and bar_sides is None and
        ss_ends is None):
        # No options, report current state and return
        indent = "  -"
        for m in structures:
            mgr = m.ribbon_xs_mgr
            print(m)
            print(indent, "helix",
                  "style=%s" % _XSInverseMap[mgr.style_helix],
                  "size=%.2g,%.2g" % mgr.scale_helix,
                  "arrow=%s" % mgr.arrow_helix,
                  "arrow size=%.2g,%.2g,%.2g,%.2g" % (mgr.scale_helix_arrow[0] +
                                                       mgr.scale_helix_arrow[1]))
            print(indent, "strand",
                  "style=%s" % _XSInverseMap[mgr.style_sheet],
                  "size=%.2g,%.2g" % mgr.scale_sheet,
                  "arrow=%s" % mgr.arrow_sheet,
                  "arrow size=%.2g,%.2g,%.2g,%.2g" % (mgr.scale_sheet_arrow[0] +
                                                        mgr.scale_sheet_arrow[1]))
            print(indent, "coil",
                  "style=%s" % _XSInverseMap[mgr.style_coil],
                  "size=%.2g,%.2g" % mgr.scale_coil)
            print(indent, "nucleic",
                  "style=%s" % _XSInverseMap[mgr.style_nucleic],
                  "size=%.2g,%.2g" % mgr.scale_nucleic)
            param = mgr.params[XSectionManager.STYLE_ROUND]
            print(indent,
                  "oval parameters:", " ".join("%s=%s" % item for item in param.items()))
            param = mgr.params[XSectionManager.STYLE_PIPING]
            print(indent,
                  "barbell parameters:", " ".join("%s=%s" % item for item in param.items()))
        return
    residues = results.atoms.residues
    is_helix = residues.is_helix
    is_sheet = residues.is_sheet
    polymer_types = residues.polymer_types
    coil_scale_changed = {}
    # Code uses half-width/thickness but command uses full width/thickness,
    # so we divide by two now so we will not need to do it multiple times
    if width is not None:
        width /= 2
    if thickness is not None:
        thickness /= 2
        # set coil parameters
    for m in structures:
        mgr = m.ribbon_xs_mgr
        if thickness is not None:
            coil_scale_changed[m] = True
            mgr.set_coil_scale(thickness, thickness)
        if (xsection is not None and
                _XSectionMap[xsection] != XSectionManager.STYLE_PIPING):
            m.ribbon_xs_mgr.set_coil_style(_XSectionMap[xsection])
    if is_helix.any():
        # set helix parameters
        for m in structures:
            mgr = m.ribbon_xs_mgr
            old_arrow_scale = None
            if width is not None or thickness is not None:
                w, h = mgr.scale_helix
                if width is not None:
                    w = width
                if thickness is not None:
                    h = thickness
                mgr.set_helix_scale(w, h)
                aw, ah = mgr.scale_helix_arrow[0]
                old_arrow_scale = aw / w
            if arrow_scale is not None or old_arrow_scale is not None:
                w, h = mgr.scale_helix
                if arrow_scale is not None:
                    aw = w * arrow_scale
                else:
                    aw = w * old_arrow_scale
                ah = h
                cw, ch = mgr.scale_coil
                mgr.set_helix_arrow_scale(aw, ah, cw, ch)
            elif coil_scale_changed.get(m, False):
                aw, ah = mgr.scale_helix_arrow[0]
                cw, ch = mgr.scale_coil
                mgr.set_helix_arrow_scale(aw, ah, cw, ch)
            if arrows_helix is not None:
                mgr.set_helix_end_arrow(arrows_helix)
            if ss_ends is not None:
                # These are the cases we deal with:
                # 1. coil->helix_start. (c_hs below)
                #    The default is coil/helix (use coil for front and helix for back).
                #    We do not change from the default because the twist from the
                #    coil does not match the twist from the helix and we must use coil/helix
                #    to look reasonable.
                # 2. helix_end->helix_start. (he_hs)
                #    Default is helix/helix.
                #    For "short", we use coil/helix.
                #    For "long" we leave it helix/helix.
                # 3. sheet_end->helix_start. (se_hs)
                #    Default is helix/helix.
                #    For "short", we use coil/helix.
                #    For "long" we use helix/helix.
                # 4. helix_end->coil. (he_c)
                #    Default is arrow/coil.
                #    For "short", we use arrow/coil.
                #    For "long", we use helix/arrow.
                # 5. helix_end->helix_start. (he_hs)
                #    Default is helix/arrow.
                #    For "short", we use arrow/coil.
                #    For "long", we use helix/arrow.
                # 6. helix_end->sheet_start. (he_ss)
                #    Default is helix/arrow.
                #    For "short", use it arrow/coil.
                #    For "long", we use helix/arrow.
                # (Defaults are defined in XSectionManager class in ribbon.py.)
                if ss_ends == "default":
                    # c_hs = (mgr.RIBBON_COIL, mgr.RIBBON_HELIX)
                    he_hs_h = (mgr.RIBBON_HELIX, mgr.RIBBON_HELIX)
                    se_hs_h = (mgr.RIBBON_HELIX, mgr.RIBBON_HELIX)
                    h_he_c = (mgr.RIBBON_HELIX_ARROW, mgr.RIBBON_COIL)
                    h_he_hs = (mgr.RIBBON_HELIX, mgr.RIBBON_HELIX_ARROW)
                    h_he_ss = (mgr.RIBBON_HELIX, mgr.RIBBON_HELIX_ARROW)
                elif ss_ends == "short":
                    # c_hs = (mgr.RIBBON_COIL, mgr.RIBBON_HELIX)
                    he_hs_h = (mgr.RIBBON_COIL, mgr.RIBBON_HELIX)
                    se_hs_h = (mgr.RIBBON_COIL, mgr.RIBBON_HELIX)
                    h_he_c = (mgr.RIBBON_HELIX_ARROW, mgr.RIBBON_COIL)
                    h_he_hs = (mgr.RIBBON_HELIX_ARROW, mgr.RIBBON_COIL)
                    h_he_ss = (mgr.RIBBON_HELIX_ARROW, mgr.RIBBON_COIL)
                elif ss_ends == "long":
                    # c_hs = (mgr.RIBBON_COIL, mgr.RIBBON_HELIX)
                    he_hs_h = (mgr.RIBBON_HELIX, mgr.RIBBON_HELIX)
                    se_hs_h = (mgr.RIBBON_HELIX, mgr.RIBBON_HELIX)
                    h_he_c = (mgr.RIBBON_HELIX, mgr.RIBBON_HELIX_ARROW)
                    h_he_hs = (mgr.RIBBON_HELIX, mgr.RIBBON_HELIX_ARROW)
                    h_he_ss = (mgr.RIBBON_HELIX, mgr.RIBBON_HELIX_ARROW)
                else:
                    raise ValueError("unexpected ss_ends value: %s" % ss_ends)
                # coil->helix_start->helix
                # mgr.set_transition(mgr.RC_COIL, mgr.RC_HELIX_START, mgr.RC_HELIX_MIDDLE, *c_hs)
                # mgr.set_transition(mgr.RC_COIL, mgr.RC_HELIX_START, mgr.RC_HELIX_END, *c_hs)
                # helix->helix_start->helix
                mgr.set_transition(mgr.RC_HELIX_END, mgr.RC_HELIX_START, mgr.RC_HELIX_MIDDLE, *he_hs_h)
                mgr.set_transition(mgr.RC_HELIX_END, mgr.RC_HELIX_START, mgr.RC_HELIX_END, *he_hs_h)
                # strand->helix_start->helix
                mgr.set_transition(mgr.RC_SHEET_END, mgr.RC_HELIX_START, mgr.RC_HELIX_MIDDLE, *se_hs_h)
                mgr.set_transition(mgr.RC_SHEET_END, mgr.RC_HELIX_START, mgr.RC_HELIX_END, *se_hs_h)
                # helix->helix_end->coil
                mgr.set_transition(mgr.RC_HELIX_START, mgr.RC_HELIX_END, mgr.RC_COIL, *h_he_c)
                mgr.set_transition(mgr.RC_HELIX_MIDDLE, mgr.RC_HELIX_END, mgr.RC_COIL, *h_he_c)
                # helix->helix_end->helix
                mgr.set_transition(mgr.RC_HELIX_START, mgr.RC_HELIX_END, mgr.RC_HELIX_START, *h_he_hs)
                mgr.set_transition(mgr.RC_HELIX_MIDDLE, mgr.RC_HELIX_END, mgr.RC_HELIX_START, *h_he_hs)
                # helix->helix_end->sheet
                mgr.set_transition(mgr.RC_HELIX_START, mgr.RC_HELIX_END, mgr.RC_SHEET_START, *h_he_ss)
                mgr.set_transition(mgr.RC_HELIX_MIDDLE, mgr.RC_HELIX_END, mgr.RC_SHEET_START, *h_he_ss)
            if xsection is not None:
                m.ribbon_xs_mgr.set_helix_style(_XSectionMap[xsection])
    if is_sheet.any():
        # set strand/sheet parameters
        for m in structures:
            mgr = m.ribbon_xs_mgr
            old_arrow_scale = None
            if width is not None or thickness is not None:
                w, h = mgr.scale_sheet
                if width is not None:
                    w = width
                if thickness is not None:
                    h = thickness
                mgr.set_sheet_scale(w, h)
                aw, ah = mgr.scale_sheet_arrow[0]
                old_arrow_scale = aw / w
            if arrow_scale is not None or old_arrow_scale is not None:
                w, h = mgr.scale_sheet
                if arrow_scale is not None:
                    aw = w * arrow_scale
                else:
                    aw = w * old_arrow_scale
                ah = h
                cw, ch = mgr.scale_coil
                mgr.set_sheet_arrow_scale(aw, ah, cw, ch)
            elif coil_scale_changed.get(m, False):
                aw, ah = mgr.scale_sheet_arrow[0]
                cw, ch = mgr.scale_coil
                mgr.set_sheet_arrow_scale(aw, ah, cw, ch)
            if arrows is not None:
                mgr.set_sheet_end_arrow(arrows)
            if ss_ends is not None:
                if ss_ends == "default":
                    # c_ss = (mgr.RIBBON_COIL, mgr.RIBBON_SHEET)
                    he_ss_s = (mgr.RIBBON_SHEET, mgr.RIBBON_SHEET)
                    se_ss_s = (mgr.RIBBON_SHEET, mgr.RIBBON_SHEET)
                    s_se_c = (mgr.RIBBON_SHEET_ARROW, mgr.RIBBON_COIL)
                    s_se_hs = (mgr.RIBBON_SHEET, mgr.RIBBON_SHEET_ARROW)
                    s_se_ss = (mgr.RIBBON_SHEET, mgr.RIBBON_SHEET_ARROW)
                elif ss_ends == "short":
                    # c_ss = (mgr.RIBBON_COIL, mgr.RIBBON_SHEET)
                    he_ss_s = (mgr.RIBBON_COIL, mgr.RIBBON_SHEET)
                    se_ss_s = (mgr.RIBBON_COIL, mgr.RIBBON_SHEET)
                    s_se_c = (mgr.RIBBON_SHEET_ARROW, mgr.RIBBON_COIL)
                    s_se_hs = (mgr.RIBBON_SHEET_ARROW, mgr.RIBBON_COIL)
                    s_se_ss = (mgr.RIBBON_SHEET_ARROW, mgr.RIBBON_COIL)
                elif ss_ends == "long":
                    # c_ss = (mgr.RIBBON_COIL, mgr.RIBBON_SHEET)
                    he_ss_s = (mgr.RIBBON_SHEET, mgr.RIBBON_SHEET)
                    se_ss_s = (mgr.RIBBON_SHEET, mgr.RIBBON_SHEET)
                    s_se_c = (mgr.RIBBON_SHEET, mgr.RIBBON_SHEET_ARROW)
                    s_se_hs = (mgr.RIBBON_SHEET, mgr.RIBBON_SHEET_ARROW)
                    s_se_ss = (mgr.RIBBON_SHEET, mgr.RIBBON_SHEET_ARROW)
                else:
                    raise ValueError("unexpected ss_ends value: %s" % ss_ends)
                # coil->sheet_start->helix
                # mgr.set_transition(mgr.RC_COIL, mgr.RC_SHEET_START, mgr.RC_SHEET_MIDDLE, *c_ss)
                # mgr.set_transition(mgr.RC_COIL, mgr.RC_SHEET_START, mgr.RC_SHEET_END, *c_ss)
                # sheet->sheet_start->helix
                mgr.set_transition(mgr.RC_HELIX_END, mgr.RC_SHEET_START, mgr.RC_SHEET_MIDDLE, *he_ss_s)
                mgr.set_transition(mgr.RC_HELIX_END, mgr.RC_SHEET_START, mgr.RC_SHEET_END, *he_ss_s)
                # sheet->sheet_start->helix
                mgr.set_transition(mgr.RC_SHEET_END, mgr.RC_SHEET_START, mgr.RC_SHEET_MIDDLE, *se_ss_s)
                mgr.set_transition(mgr.RC_SHEET_END, mgr.RC_SHEET_START, mgr.RC_SHEET_END, *se_ss_s)
                # sheet->sheet_end->coil
                mgr.set_transition(mgr.RC_SHEET_START, mgr.RC_SHEET_END, mgr.RC_COIL, *s_se_c)
                mgr.set_transition(mgr.RC_SHEET_MIDDLE, mgr.RC_SHEET_END, mgr.RC_COIL, *s_se_c)
                # sheet->sheet_end->helix
                mgr.set_transition(mgr.RC_SHEET_START, mgr.RC_SHEET_END, mgr.RC_HELIX_START, *s_se_hs)
                mgr.set_transition(mgr.RC_SHEET_MIDDLE, mgr.RC_SHEET_END, mgr.RC_HELIX_START, *s_se_hs)
                # sheet->sheet_end->sheet
                mgr.set_transition(mgr.RC_SHEET_START, mgr.RC_SHEET_END, mgr.RC_SHEET_START, *s_se_ss)
                mgr.set_transition(mgr.RC_SHEET_MIDDLE, mgr.RC_SHEET_END, mgr.RC_SHEET_START, *s_se_ss)
            if xsection is not None:
                m.ribbon_xs_mgr.set_sheet_style(_XSectionMap[xsection])
    if (polymer_types == Residue.PT_NUCLEIC).any():
        # set nucleic parameters
        for m in structures:
            mgr = m.ribbon_xs_mgr
            if width is not None or thickness is not None:
                w, h = mgr.scale_nucleic
                # Invert width and thickness since nucleic cross section
                # is perpendicular to protein cross section
                if width is not None:
                    h = width
                if thickness is not None:
                    w = thickness
                mgr.set_nucleic_scale(w, h)
            if xsection is not None:
                m.ribbon_xs_mgr.set_nucleic_style(_XSectionMap[xsection])
    # process sides, bar_sides and bar_scale
    oval_params = {}
    bar_params = {}
    if sides is not None:
        oval_params["sides"] = sides
    if oval_params:
        for m in structures:
            m.ribbon_xs_mgr.set_params(XSectionManager.STYLE_ROUND, **oval_params)
    if bar_scale is not None:
        bar_params["ratio"] = bar_scale
    if bar_sides is not None:
        bar_params["sides"] = bar_sides
    if bar_params:
        for m in structures:
            m.ribbon_xs_mgr.set_params(XSectionManager.STYLE_PIPING, **bar_params)


# Other command functions (to be removed)


def cartoon_xsection(session, structures=None, helix=None, strand=None, coil=None, nucleic=None):
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
    if helix is None and strand is None and coil is None and nucleic is None:
        for m in _get_structures(session, structures):
            mgr = m.ribbon_xs_mgr
            print("%s: helix=%s strand=%s coil=%s nucleic=%s" % (m, _XSInverseMap[mgr.style_helix],
                                                                 _XSInverseMap[mgr.style_sheet],
                                                                 _XSInverseMap[mgr.style_coil],
                                                                 _XSInverseMap[mgr.style_nucleic]))
        return
    for m in _get_structures(session, structures):
        if helix is not None:
            m.ribbon_xs_mgr.set_helix_style(_XSectionMap[helix])
        if strand is not None:
            m.ribbon_xs_mgr.set_sheet_style(_XSectionMap[strand])
        if coil is not None:
            m.ribbon_xs_mgr.set_coil_style(_XSectionMap[coil])
        if nucleic is not None:
            m.ribbon_xs_mgr.set_nucleic_style(_XSectionMap[nucleic])


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
    if (helix is None and arrow_helix is None and strand is None and arrow_strand is None
        and coil is None and nucleic is None):
        for m in _get_structures(session, structures):
            mgr = m.ribbon_xs_mgr
            print("%s:" % m,
                  "helix=%.2g,%.2g" % mgr.scale_helix,
                  "arrow_helix=%.2g,%.2g,%.2g,%.2g" % (mgr.scale_helix_arrow[0] +
                                                       mgr.scale_helix_arrow[1]),
                  "strand=%.2g,%.2g" % mgr.scale_sheet,
                  "arrow_strand=%.2g,%.2g,%.2g,%.2g" % (mgr.scale_sheet_arrow[0] +
                                                        mgr.scale_sheet_arrow[1]),
                  "coil=%.2g,%.2g" % mgr.scale_coil,
                  "nucleic=%.2g,%.2g" % mgr.scale_nucleic)
        return
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
    elif linker == "short2":
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
    if helix is None and strand is None:
        for m in _get_structures(session, structures):
            mgr = m.ribbon_xs_mgr
            print("%s: helix arrow=%s strand arrow=%s" % (m, mgr.arrow_helix,
                                                             mgr.arrow_sheet))
        return
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
    if faceted is None and sides is None:
        for m in _get_structures(session, structures):
            mgr = m.ribbon_xs_mgr
            param = mgr.params[XSectionManager.STYLE_ROUND]
            print("%s: %s" % (m, ", ".join("%s: %s" % item for item in param.items())))
        return
    params = {}
    if faceted is not None:
        params["faceted"] = faceted
    if sides is not None:
        params["sides"] = sides
    if params:
        for m in _get_structures(session, structures):
            m.ribbon_xs_mgr.set_params(XSectionManager.STYLE_ROUND, **params)


def cartoon_param_piping(session, structures=None, faceted=None, sides=None, ratio=None):
    '''Set cartoon piping ribbon parameters for specified structures.

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
    if faceted is None and sides is None and ratio is None:
        for m in _get_structures(session, structures):
            mgr = m.ribbon_xs_mgr
            param = mgr.params[XSectionManager.STYLE_PIPING]
            print("%s: %s" % (m, ", ".join("%s: %s" % item for item in param.items())))
        return
    params = {}
    if faceted is not None:
        params["faceted"] = faceted
    if sides is not None:
        params["sides"] = sides
    if ratio is not None:
        params["ratio"] = ratio
    if params:
        for m in _get_structures(session, structures):
            m.ribbon_xs_mgr.set_params(XSectionManager.STYLE_PIPING, **params)


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


class EvenIntArg(Annotation):
    """Annotation for even integers (for "sides")"""
    name = "an even integer"

    @classmethod
    def parse(cls, text, session):
        from chimerax.core.commands import IntArg
        try:
            token, text, rest = IntArg.parse(text, session)
        except AnnotationError:
            raise AnnotationError("Expected %s" % cls.name)
        if (token % 2) == 1:
            raise AnnotationError("Expected %s" % cls.name)
        return token, text, rest


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
                                ("sides", Bounded(IntArg, 3, 24)),
                                ("opacity", Bounded(FloatArg, 0.0, 1.0)),
                                ],
                       synopsis='set cartoon tether options for specified structures')
        register(command_name + " tether", desc, cartoon_tether)

        desc = CmdDesc(optional=[("spec", AtomSpecArg)],
                       keyword=[("width", FloatArg),
                                ("thickness", FloatArg),
                                ("arrows", BoolArg),
                                ("arrows_helix", BoolArg),
                                ("arrow_scale", Bounded(FloatArg, 1.0, 3.0)),
                                ("xsection", EnumOf(_XSectionMap.keys())),
                                ("sides", Bounded(EvenIntArg, 3, 24)),
                                ("bar_scale", FloatArg),
                                ("bar_sides", Bounded(EvenIntArg, 3, 24)),
                                ("ss_ends", EnumOf(["default", "short", "long"])),
                                # ("cylinders", BoolArg),
                                ],
                       synopsis='set cartoon style for secondary structures in specified models')
        register(command_name + " style", desc, cartoon_style)

        # Other command registrations (to be removed)

        xs = EnumOf(_XSectionMap.keys())
        desc = CmdDesc(optional=[("structures", AtomicStructuresArg)],
                       keyword=[("helix", xs),
                                ("strand", xs),
                                ("coil", xs),
                                ("nucleic", xs),
                                ],
                       synopsis='set cartoon cross section options for specified structures')
        register(command_name + " xsection", desc, cartoon_xsection)

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
                                ("ratio", Bounded(FloatArg, 0.2, 1.0)),
                                ],
                       synopsis='set cartoon piping ribbon options for specified structures')
        register(command_name + " param piping", desc, cartoon_param_piping)
