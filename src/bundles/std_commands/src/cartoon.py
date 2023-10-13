# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2022 Regents of the University of California. All rights reserved.
# This software is provided pursuant to the ChimeraX license agreement, which
# covers academic and commercial uses. For more information, see
# <http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
#
# This file is part of the ChimeraX library. You can also redistribute and/or
# modify it under the GNU Lesser General Public License version 2.1 as
# published by the Free Software Foundation. For more details, see
# <https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html>
#
# This file is distributed WITHOUT ANY WARRANTY; without even the implied
# warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. This notice
# must be embedded in or attached to all copies, including partial copies, of
# the software or any revisions or derivations thereof.
# === UCSF ChimeraX Copyright ===

from chimerax.atomic.ribbon import XSectionManager
from chimerax.atomic import Residue, Structure
from chimerax.core.commands import Annotation, AnnotationError

_TetherShapeMap = {
    "cone": Structure.TETHER_CONE,
    "cylinder": Structure.TETHER_CYLINDER,
    "steeple": Structure.TETHER_REVERSE_CONE,
}
_TetherShapeInverseMap = dict([(v, k) for k, v in _TetherShapeMap.items()])
_XSectionMap = {
    "rectangle": XSectionManager.STYLE_SQUARE,
    "square": XSectionManager.STYLE_SQUARE,
    "oval": XSectionManager.STYLE_ROUND,
    "round": XSectionManager.STYLE_ROUND,
    "barbell": XSectionManager.STYLE_PIPING,
    "piping": XSectionManager.STYLE_PIPING,
}
_XSectionInverseMap = dict([(v, k) for k, v in _XSectionMap.items()])
_ModeHelixMap = {
    "default": Structure.RIBBON_MODE_DEFAULT,
    "tube": Structure.RIBBON_MODE_ARC,
    "wrap": Structure.RIBBON_MODE_WRAP,
}
_ModeHelixInverseMap = dict([(v, k) for k, v in _ModeHelixMap.items()])
_ModeStrandMap = {
    "default": Structure.RIBBON_MODE_DEFAULT,
    "plank": Structure.RIBBON_MODE_ARC,
}
_ModeStrandInverseMap = dict([(v, k) for k, v in _ModeStrandMap.items()])


def cartoon(session, atoms=None, smooth=None, suppress_backbone_display=None, spine=False):
    '''Display cartoon for specified residues.

    Parameters
    ----------
    atoms : Atoms
        Show ribbons for the specified residues. If no atom specifier is given then ribbons are shown
        for all residues.  Residues that are already shown as ribbons remain shown as ribbons.
    smooth : floating point number
        Adjustment factor for strand and helix smoothing.  A factor of zero means the
        cartoon will pass through the atom position.  A factor of one means the cartoon
        will pass through the "ideal" position, e.g., center of the cylinder that best
        fits a helix.  A factor of "default" means to return to default (0.7 for strands
        and 0 for everything else).
    suppress_backbone_display : boolean
        Set whether displaying a ribbon hides the sphere/ball/stick representation of
        backbone atoms.
    spine : boolean
        Display ribbon "spine" (horizontal lines across center of ribbon).
        This parameter applies at the atomic structure level, so setting it for any residue
        sets it for the entire structure.
    '''
    if atoms is None:
        from chimerax.atomic import all_residues
        residues = all_residues(session)
    else:
        residues = atoms.unique_residues
    from chimerax.core.undo import UndoState
    undo_state = UndoState("cartoon")
    undo_state.add(residues, "ribbon_displays", residues.ribbon_displays, True)
    residues.ribbon_displays = True
    if smooth is not None:
        if smooth == "default":
            # Convert to C++ default value
            smooth = -1.0
        undo_state.add(residues, "ribbon_adjusts", residues.ribbon_adjusts, smooth)
        residues.ribbon_adjusts = smooth
    if suppress_backbone_display is not None:
        undo_state.add(residues, "ribbon_hide_backbones",
                       residues.ribbon_hide_backbones, suppress_backbone_display)
        residues.ribbon_hide_backbones = suppress_backbone_display
    if spine is not None:
        structures = residues.unique_structures
        undo_state.add(structures, "ribbon_show_spines", structures.ribbon_show_spines, spine)
        structures.ribbon_show_spines = spine
    session.undo.register(undo_state)


def _get_structures(session, structures):
    if structures is None or structures is True:
        # True is the NoArg case
        from chimerax.atomic import all_atomic_structures, AtomicStructures
        structures = AtomicStructures(all_atomic_structures(session))
        
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
    if scale is None and shape is None and sides is None and opacity is None:
        indent = "  -"
        for m in structures:
            print(m)
            print(indent, "scales", m.ribbon_tether_scale)
            print(indent, "shapes", _TetherShapeInverseMap[m.ribbon_tether_shape])
            print(indent, "sides", m.ribbon_tether_sides)
            print(indent, "opacity", m.ribbon_tether_opacity)
        return
    from chimerax.core.undo import UndoState
    undo_state = UndoState("cartoon tether")
    if scale is not None:
        undo_state.add(structures, "ribbon_tether_scales", structures.ribbon_tether_scales, scale)
        structures.ribbon_tether_scales = scale
    if shape is not None:
        ts = _TetherShapeMap.get(shape, Structure.TETHER_CONE)
        undo_state.add(structures, "ribbon_tether_shapes", structures.ribbon_tether_shapes, ts)
        structures.ribbon_tether_shapes = ts
    if sides is not None:
        undo_state.add(structures, "ribbon_tether_sides", structures.ribbon_tether_sides, sides)
        structures.ribbon_tether_sides = sides
    if opacity is not None:
        undo_state.add(structures, "ribbon_tether_opacities", structures.ribbon_tether_opacities, opacity)
        structures.ribbon_tether_opacities = opacity
    session.undo.register(undo_state)


def cartoon_style(session, atoms=None, width=None, thickness=None, arrows=None, arrows_helix=None,
                  arrow_scale=None, xsection=None, sides=None,
                  bar_scale=None, bar_sides=None, ss_ends=None,
                  mode_helix=None, mode_strand=None, radius=None,
                  divisions=None, spline_normals=None):
    '''Set cartoon style options for secondary structures in specified structures.

    Parameters
    ----------
    atoms : Atoms
        Set style for all secondary structure types that include the specified residues.
        If no atoms are given then style is set for all secondary structure types.
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
    divisions : integer
        Number of segments per residue
    bar_scale : floating point number
        Ratio of barbell connector to ends.
    bar_sides : integer
        Number of sides for barbell cross sections.
    ss_ends : string
        Length of helix/strand representation relative to backbone atoms.
        One of "default", "short" or "long".
    mode_helix : string
        Choose how helices are rendered.
        "default" uses ribbons through the alpha carbons.
        "tube" uses a tube along an arc so that the alpha carbons are on the surface of the tube.
    mode_strand : string
        Same argument values are mode_helix.
    radius: floating point number
        Radius of helices as cylinders
    '''
    if atoms is None:
        from chimerax.atomic import all_residues, all_structures
        residues = all_residues(session)
        structures = all_structures(session)
    else:
        residues = atoms.unique_residues
        structures = residues.unique_structures
    import inspect
    argvalues = inspect.getargvalues(inspect.currentframe())
    for name in argvalues.args:
        if name in ("session", "atoms"):
            continue
        if argvalues.locals.get(name, None) is not None:
            no_values = False
            break
    else:
        no_values = True
    if no_values:
    # if (width is None and thickness is None and arrows is None and
    #     arrows_helix is None and arrow_scale is None and xsection is None and
    #     sides is None and divisions is None and
    #     bar_scale is None and bar_sides is None and
    #     ss_ends is None and mode_helix is None and mode_strand is None and
    #     radius is None and spline_normals is None):
        # No options, report current state and return
        indent = "  -"
        for m in structures:
            mgr = m.ribbon_xs_mgr
            print(m)
            print(indent, "helix",
                  "mode=%s" % _ModeHelixInverseMap[m.ribbon_mode_helix],
                  "xsection=%s" % _XSectionInverseMap[mgr.style_helix],
                  "width=%.2g" % (mgr.scale_helix[0] * 2),
                  "height=%.2g" % (mgr.scale_helix[1] * 2),
                  "arrow=%s" % mgr.arrow_helix,
                  "arrow scale=%.2g" % (mgr.scale_helix_arrow[0][0] / mgr.scale_helix[0]))
            print(indent, "strand",
                  "mode=%s" % _ModeStrandInverseMap[m.ribbon_mode_strand],
                  "xsection=%s" % _XSectionInverseMap[mgr.style_sheet],
                  "width=%.2g" % (mgr.scale_sheet[0] * 2),
                  "height=%.2g" % (mgr.scale_sheet[1] * 2),
                  "arrow=%s" % mgr.arrow_sheet,
                  "arrow scale=%.2g" % (mgr.scale_sheet_arrow[0][0] / mgr.scale_sheet[0]))
            print(indent, "coil",
                  "xsection=%s" % _XSectionInverseMap[mgr.style_coil],
                  "width=%.2g" % (mgr.scale_coil[0] * 2),
                  "height=%.2g" % (mgr.scale_coil[1] * 2))
            print(indent, "nucleic",
                  "xsection=%s" % _XSectionInverseMap[mgr.style_nucleic],
                  "width=%.2g" % (mgr.scale_nucleic[0] * 2),
                  "height=%.2g" % (mgr.scale_nucleic[1] * 2))
            from chimerax.atomic.structure import structure_graphics_updater
            lod = structure_graphics_updater(session).level_of_detail
            print(indent, "divisions=%d" % lod.ribbon_divisions(m.num_ribbon_residues))
            param = mgr.params[XSectionManager.STYLE_ROUND]
            print(indent, "oval parameters:",
                  "sides=%d" % param["sides"])
            param = mgr.params[XSectionManager.STYLE_PIPING]
            print(indent, "barbell parameters:",
                  "sides=%d" % param["sides"],
                  "scale=%.2g" % param["ratio"])
        return
    is_helix = residues.is_helix
    is_strand = residues.is_strand
    polymer_types = residues.polymer_types
    from numpy import logical_and, logical_not
    is_coil = logical_and(logical_and(logical_not(is_helix), logical_not(is_strand)),
                          polymer_types == Residue.PT_PROTEIN)
    coil_scale_changed = {}
    # Code uses half-width/thickness but command uses full width/thickness,
    # so we divide by two now so we will not need to do it multiple times
    if width is not None:
        width /= 2
    if thickness is not None:
        thickness /= 2
    from chimerax.core.undo import UndoState
    undo_state = UndoState("cartoon style")
    if is_coil.any():
        # set coil parameters
        for m in structures:
            mgr = m.ribbon_xs_mgr
            if thickness is not None:
                coil_scale_changed[m] = True
                undo_state.add(mgr, "set_coil_scale", mgr.scale_coil, (thickness, thickness), "MA")
                mgr.set_coil_scale(thickness, thickness)
            if (xsection is not None and
                    _XSectionMap[xsection] != XSectionManager.STYLE_PIPING):
                undo_state.add(mgr, "set_coil_style", mgr.style_coil, _XSectionMap[xsection], "M")
                mgr.set_coil_style(_XSectionMap[xsection])
    if is_helix.any():
        # set helix parameters
        for m in structures:
            mgr = m.ribbon_xs_mgr
            old_arrow_scale = None
            if width is not None or thickness is not None:
                w, h = mgr.scale_helix
                aw, ah = mgr.scale_helix_arrow[0]
                old_arrow_scale = aw / w
                if width is not None:
                    w = width
                if thickness is not None:
                    h = thickness
                undo_state.add(mgr, "set_helix_scale", mgr.scale_helix, (w, h), "MA")
                mgr.set_helix_scale(w, h)
            if arrow_scale is not None or old_arrow_scale is not None:
                w, h = mgr.scale_helix
                if arrow_scale is not None:
                    aw = w * arrow_scale
                else:
                    aw = w * old_arrow_scale
                ah = h
                cw, ch = mgr.scale_coil
                old = mgr.scale_helix_arrow[0] + mgr.scale_helix_arrow[1]
                undo_state.add(mgr, "set_helix_arrow_scale", old, (aw, ah, cw, ch), "MA")
                mgr.set_helix_arrow_scale(aw, ah, cw, ch)
            elif coil_scale_changed.get(m, False):
                aw, ah = mgr.scale_helix_arrow[0]
                cw, ch = mgr.scale_coil
                old = mgr.scale_helix_arrow[0] + mgr.scale_helix_arrow[1]
                undo_state.add(mgr, "set_helix_arrow_scale", old, (aw, ah, cw, ch), "MA")
                mgr.set_helix_arrow_scale(aw, ah, cw, ch)
            if arrows_helix is not None:
                undo_state.add(mgr, "set_helix_end_arrow", mgr.arrow_helix, arrows_helix, "M")
                mgr.set_helix_end_arrow(arrows_helix)
            if ss_ends is not None:
                # TODO: save undo data
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
                undo_state.add(mgr, "set_helix_style", mgr.style_helix, _XSectionMap[xsection], "M")
                mgr.set_helix_style(_XSectionMap[xsection])
    if is_strand.any():
        # set strand/sheet parameters
        for m in structures:
            mgr = m.ribbon_xs_mgr
            old_arrow_scale = None
            if width is not None or thickness is not None:
                w, h = mgr.scale_sheet
                aw, ah = mgr.scale_sheet_arrow[0]
                old_arrow_scale = aw / w
                if width is not None:
                    w = width
                if thickness is not None:
                    h = thickness
                undo_state.add(mgr, "set_sheet_scale", mgr.scale_sheet, (w, h), "MA")
                mgr.set_sheet_scale(w, h)
            if arrow_scale is not None or old_arrow_scale is not None:
                w, h = mgr.scale_sheet
                if arrow_scale is not None:
                    aw = w * arrow_scale
                else:
                    aw = w * old_arrow_scale
                ah = h
                cw, ch = mgr.scale_coil
                old = mgr.scale_sheet_arrow[0] + mgr.scale_sheet_arrow[1]
                undo_state.add(mgr, "set_sheet_arrow_scale", old, (aw, ah, cw, ch), "MA")
                mgr.set_sheet_arrow_scale(aw, ah, cw, ch)
            elif coil_scale_changed.get(m, False):
                aw, ah = mgr.scale_sheet_arrow[0]
                cw, ch = mgr.scale_coil
                old = mgr.scale_sheet_arrow[0] + mgr.scale_sheet_arrow[1]
                undo_state.add(mgr, "set_sheet_arrow_scale", old, (aw, ah, cw, ch), "MA")
                mgr.set_sheet_arrow_scale(aw, ah, cw, ch)
            if arrows is not None:
                undo_state.add(mgr, "set_sheet_end_arrow", mgr.arrow_sheet, arrows, "M")
                mgr.set_sheet_end_arrow(arrows)
            if ss_ends is not None:
                # TODO: save undo data
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
                undo_state.add(mgr, "set_sheet_style", mgr.style_helix, _XSectionMap[xsection], "M")
                mgr.set_sheet_style(_XSectionMap[xsection])
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
                undo_state.add(mgr, "set_nucleic_scale", mgr.scale_nucleic, (w, h), "MA")
                mgr.set_nucleic_scale(w, h)
            if xsection is not None:
                undo_state.add(mgr, "set_nucleic_style", mgr.style_nucleic, _XSectionMap[xsection], "M")
                mgr.set_nucleic_style(_XSectionMap[xsection])
    # process sides, bar_sides and bar_scale
    oval_params = {}
    bar_params = {}
    if sides is not None:
        oval_params["sides"] = sides
    if oval_params:
        for m in structures:
            mgr = m.ribbon_xs_mgr
            old_param = mgr.params[XSectionManager.STYLE_ROUND].copy()
            old_param["style"] = XSectionManager.STYLE_ROUND
            mgr.set_params(XSectionManager.STYLE_ROUND, **oval_params)
            new_param = mgr.params[XSectionManager.STYLE_ROUND].copy()
            new_param["style"] = XSectionManager.STYLE_ROUND
            undo_state.add(mgr, "set_params", old_param, new_param, "MK")
    if bar_scale is not None:
        bar_params["ratio"] = bar_scale
    if bar_sides is not None:
        bar_params["sides"] = bar_sides
    if bar_params:
        for m in structures:
            mgr = m.ribbon_xs_mgr
            old_param = mgr.params[XSectionManager.STYLE_PIPING].copy()
            old_param["style"] = XSectionManager.STYLE_PIPING
            mgr.set_params(XSectionManager.STYLE_PIPING, **bar_params)
            new_param = mgr.params[XSectionManager.STYLE_PIPING].copy()
            new_param["style"] = XSectionManager.STYLE_PIPING
            undo_state.add(mgr, "set_params", old_param, new_param, "MK")
    # process divisions which is actually handled by level of detail
    if divisions is not None:
        from chimerax.atomic.structure import structure_graphics_updater
        gu = structure_graphics_updater(session)
        prev_divisions = gu.level_of_detail.ribbon_fixed_divisions
        undo_state.add(gu, "set_ribbon_divisions", prev_divisions, divisions, option = 'M')
        gu.set_ribbon_divisions(divisions)
    # process modes
    if mode_helix is not None:
        mode = _ModeHelixMap.get(mode_helix, None)
        for m in structures:
            undo_state.add(m, "ribbon_mode_helix", m.ribbon_mode_helix, mode)
            m.ribbon_mode_helix = mode
    if mode_strand is not None:
        mode = _ModeStrandMap.get(mode_strand, None)
        for m in structures:
            undo_state.add(m, "ribbon_mode_strand", m.ribbon_mode_strand, mode)
            m.ribbon_mode_strand = mode
    # process radius
    if radius is not None:
        if radius == "auto":
            radius = None
        for m in structures:
            mgr = m.ribbon_xs_mgr
            undo_state.add(mgr, "set_tube_radius", mgr.tube_radius, radius, "M")
            mgr.set_tube_radius(radius)
    if spline_normals is not None:
        for m in structures:
            undo_state.add(m, "spline_normals", m.spline_normals, spline_normals)
            m.spline_normals = spline_normals
    session.undo.register(undo_state)


def uncartoon(session, atoms=None):
    '''Undisplay ribbons for specified residues.

    Parameters
    ----------
    atoms : Atoms
        Hide ribbons for the specified residues. If no atoms are given then all ribbons are hidden.
    '''
    if atoms is None:
        from chimerax.atomic import all_residues
        residues = all_residues(session)
    else:
        residues = atoms.unique_residues
    from chimerax.core.undo import UndoState
    undo_state = UndoState("cartoon hide")
    undo_state.add(residues, "ribbon_displays", residues.ribbon_displays, False)
    residues.ribbon_displays = False
    session.undo.register(undo_state)

# -----------------------------------------------------------------------------
#
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

# -----------------------------------------------------------------------------
#
def register_command(logger):
    from chimerax.core.commands import register, Or, CmdDesc
    from chimerax.core.commands import Bounded, FloatArg, PositiveFloatArg, EnumOf, BoolArg, IntArg, TupleOf, NoArg
    from chimerax.atomic import AtomsArg, AtomicStructuresArg
    desc = CmdDesc(optional=[("atoms", AtomsArg)],
                   keyword=[("smooth", Or(Bounded(FloatArg, 0.0, 1.0),
                                          EnumOf(["default"]))),
                            ("suppress_backbone_display", BoolArg),
                            ("spine", BoolArg),
                            ],
                   hidden=["spine"],
                   synopsis='display cartoon for specified residues')
    register("cartoon", desc, cartoon, logger=logger)

    desc = CmdDesc(optional=[("structures", AtomicStructuresArg)],
                   keyword=[("scale", Bounded(FloatArg, 0.0, 1.0)),
                            ("shape", EnumOf(_TetherShapeMap.keys())),
                            ("sides", Bounded(IntArg, 3, 24)),
                            ("opacity", Bounded(FloatArg, 0.0, 1.0)),
                            ],
                   synopsis='set cartoon tether options for specified structures')
    register("cartoon tether", desc, cartoon_tether, logger=logger)

    desc = CmdDesc(optional=[("atoms", AtomsArg)],
                   keyword=[("width", PositiveFloatArg),
                            ("thickness", PositiveFloatArg),
                            ("arrows", BoolArg),
                            ("arrows_helix", BoolArg),
                            ("arrow_scale", Bounded(FloatArg, 1.0, 5.0)),
                            ("xsection", EnumOf(_XSectionMap.keys())),
                            ("sides", Bounded(EvenIntArg, 3, 24)),
                            ("divisions", Bounded(IntArg, 2, 40)),
                            ("bar_scale", PositiveFloatArg),
                            ("bar_sides", Bounded(EvenIntArg, 3, 24)),
                            ("ss_ends", EnumOf(["default", "short", "long"])),
                            ("mode_helix", EnumOf(list(_ModeHelixMap.keys()))),
                            ("mode_strand", EnumOf(list(_ModeStrandMap.keys()))),
                            ("radius", Or(PositiveFloatArg, EnumOf(["auto"]))),
                            ("spline_normals", BoolArg),
                            ],
                   hidden=["ss_ends", "mode_strand", "spline_normals"],
                   synopsis='set cartoon style for secondary structures in specified models')
    register("cartoon style", desc, cartoon_style, logger=logger)
    desc = CmdDesc(optional=[("atoms", AtomsArg)],
                   synopsis='undisplay cartoon for specified residues')
    register("cartoon hide", desc, uncartoon, logger=logger)
    from chimerax.core.commands import create_alias
    create_alias("~cartoon", "cartoon hide $*", logger=logger)
