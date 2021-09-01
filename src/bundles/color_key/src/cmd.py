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

auto_color_strings = ['default', 'auto']

from chimerax.core.errors import UserError
def key_cmd(session, colors_and_labels=None, *, pos=None, size=None, font_size=None, bold=None, italic=None,
        color_treatment=None, justification=None, label_side=None, numeric_label_spacing=None,
        label_color=None, label_offset=None, font=None, border=None, border_color=None, border_width=None,
        show_tool=False, ticks=None, tick_length=None, tick_thickness=None):
    if colors_and_labels is not None:
        # list of (color, label) and/or (colormap, label1, label2...) items.  Convert...
        from chimerax.core.colors import Colormap
        rgbas_and_labels = []
        for item in colors_and_labels:
            if isinstance(item[0], Colormap):
                cmap, *labels = item
                if len(cmap.colors) != len(labels):
                    raise ValueError("Must supply the same number of labels as colors in the colormap")
                rgbas_and_labels.extend(list(zip(cmap.colors, labels)))
            else:
                rgbas_and_labels.append((item[0].rgba, item[1]))
        if len(rgbas_and_labels) < 2:
            raise UserError("Must specify at least two colors for key")
    from .model import get_model
    key = get_model(session, create=False)
    if key is None:
        key = get_model(session)
    if pos is not None:
        key.pos = pos
    if size is not None:
        key.size = size
    if font_size is not None:
        key.font_size = font_size
    if bold is not None:
        key.bold = bold
    if italic is not None:
        key.italic = italic
    if color_treatment is not None:
        key.color_treatment = color_treatment
    if justification is not None:
        key.justification = justification
    if label_side is not None:
        key.label_side = label_side
    if numeric_label_spacing is not None:
        key.numeric_label_spacing = numeric_label_spacing
    if label_color is not None:
        if label_color in auto_color_strings:
            key.label_rgba = None
        else:
            key.label_rgba = label_color.rgba
    if label_offset is not None:
        key.label_offset = label_offset
    if font is not None:
        key.font = font
    if border is not None:
        key.border = border
    if border_color is not None:
        if border_color in auto_color_strings:
            key.border_rgba = None
        else:
            key.border_rgba = border_color.rgba
    if border_width is not None:
        key.border_width = border_width
    if ticks is not None:
        key.ticks = ticks
    if tick_length is not None:
        key.tick_length = tick_length
    if tick_thickness is not None:
        key.tick_thickness = tick_thickness
    if pos is not None or size is not None:
        if key.pos[0] < 0 or key.pos[1] < 0 \
        or (key.pos[0] + key.size[0]) > 1 or (key.pos[1] + key.size[1]) > 1:
            session.logger.warning("Key is partially or completely offscreen")
    if colors_and_labels is not None:
        key.rgbas_and_labels = rgbas_and_labels
    if show_tool and session.ui.is_gui and not session.in_script:
        from chimerax.core.commands import run
        run(session,"ui tool show 'Color Key'", log=False)
    return key

def key_delete_cmd(session):
    from .model import get_model
    key = get_model(session, create=False)
    if key is not None:
        key.delete()

def palette_equal(p1, p2):
    if len(p1) != len(p2):
        return False
    tolerance = 1 / 512
    def len4(c):
        if len(c) == 4:
            return c
        else:
            return [x for x in c] + [1.0]
    for c1, c2 in zip(p1, p2):
        for v1, v2 in zip(len4(c1), len4(c2)):
            if abs(v1 - v2) > tolerance:
                return False
    return True

def palette_name(rgbas):
    from chimerax.core.colors import BuiltinColormaps
    for name, cm in BuiltinColormaps.items():
        if palette_equal(cm.colors, rgbas):
            return name
    return None

def _precision_values(values, precision):
    if precision is None:
        return "%g", values

    # if all integer, return as is
    for v in values:
        if int(v) != v:
            break
    else:
        return "%d", values

    # how many non-zero digits to the left of the decimal point?
    int_digits = 0
    for v in values:
        iv = int(v+0.5)
        if iv > 0:
            int_digits = max(int_digits, len(str(iv)))

    if int_digits >= precision:
        return "%d", [int(v+0.5) for v in values]

    # use number of decimal places implied by remaining precision,
    # but reduce that if last place of all values is zero
    for decimal_places in range(precision - int_digits, 0, -1):
        fmt = "%%.%df" % decimal_places
        for v in values:
            if (fmt % v)[-1] != '0':
                break
        else:
            continue
        return fmt, values
    # should only happen if values very close to ints
    return "%d", [int(v+0.5) for v in values]

def show_key(session, color_map, *, show_tool=True, precision=3):
    """If precision is None, use full precision"""
    from chimerax.core.commands import run, StringArg
    from chimerax.core.colors import color_name, rgba_to_rgba8
    palette = palette_name(color_map.colors)
    v_fmt, values = _precision_values(color_map.data_values, precision)
    if palette is None:
        key_arg = ' '.join([StringArg.unparse(("%s:" + v_fmt) % (color_name(rgba_to_rgba8(c)), dv))
            for c, dv in zip(color_map.colors, values)])
    else:
        key_arg = "%s %s" % (StringArg.unparse(palette), " ".join([(':' + v_fmt) % dv for dv in values]))
    if show_tool:
        key_arg += " showTool true"
    run(session, "key " + key_arg)

from chimerax.core.commands import Annotation, ColorArg, StringArg, AnnotationError, next_token, \
    ColormapArg, Or
class ColorLabelPairArg(Annotation):
    name = "color:label pair"

    @staticmethod
    def parse(text, session):
        if not text:
            raise AnnotationError("Expected %s" % ColorLabelPairArg.name)
        token, text, rest = next_token(text)
        while ':' not in token:
            if not rest.lstrip():
                raise AnnotationError("No ':' found")
            token2, text2, rest = next_token(rest.lstrip())
            token += ' ' + token2
            text += ' ' + text2

        color_token, label_token = token.split(':', 1)
        if not color_token:
            raise AnnotationError("No color before ':' in %s" % ColorLabelPairArg.name)
        if label_token:
            label = label_token
        else:
            label = None
        color, ignore, color_rest = ColorArg.parse(color_token, session)
        if color_rest:
            raise AnnotationError("Trailing text after color '%s'" % color_token)
        return (color, label), text, rest

    @staticmethod
    def unparse(value, session=None):
        rgba, label = value
        from chimerax.core.colors import Color
        text = ColorArg.unparse(Color(rgba), session)
        if label:
            text += ':' + label
        return text

class PaletteLabelsArg(Annotation):
    name = "palette-name :label1 :label2 ..."

    @staticmethod
    def parse(text, session):
        if not text:
            raise AnnotationError("Expected %s" % PaletteLabelsArg.name)
        cmap, text, rest = ColormapArg.parse(text, session)
        if len(cmap.colors) < 2:
            raise AnnotationError("Palette must contain at least two colors")
        final_text = text
        vals = [cmap]
        num_labels = len(cmap.colors)
        for i in range(num_labels):
            if not rest.lstrip():
                raise AnnotationError("Need at least %d labels to match palette" % num_labels)
            label_token, text, rest = next_token(rest.lstrip(), session)
            final_text += ' ' + text
            if not label_token.startswith(':'):
                raise AnnotationError("Each label must be prefixed with ':'")
            vals.append(label_token[1:])
        return vals, final_text, rest

    @staticmethod
    def unparse(value, session=None):
        cmap, *labels = value
        text = ColormapArg.unparse(cmap, session)
        if labels:
            text += ' ' + ' '.join([StringArg.unparse(':' + l, session) for l in labels])
        return text

class RepeatableOr(Or):
    allow_repeat = True

def register_command(logger):
    from chimerax.core.commands import CmdDesc, register, Float2Arg, TupleOf, BoolArg, \
        PositiveIntArg, StringArg, EnumOf, FloatArg, Or, create_alias, NonNegativeFloatArg
    from .model import ColorKeyModel
    cmd_desc = CmdDesc(
        optional=[('colors_and_labels', RepeatableOr(ColorLabelPairArg, PaletteLabelsArg))],
        keyword=[
            ('bold', BoolArg),
            ('border', BoolArg),
            ('border_color', Or(EnumOf(auto_color_strings), ColorArg)),
            ('border_width', NonNegativeFloatArg),
            ('color_treatment', EnumOf([x.split()[0] for x in ColorKeyModel.color_treatments])),
            ('font', StringArg),
            ('font_size', PositiveIntArg),
            ('italic', BoolArg),
            ('justification', EnumOf([x.split()[0] for x in ColorKeyModel.justifications])),
            ('label_color', Or(EnumOf(auto_color_strings), ColorArg)),
            ('label_offset', FloatArg),
            ('label_side', EnumOf([x.split()[0] for x in ColorKeyModel.label_sides])),
            ('numeric_label_spacing', EnumOf([x.split()[0] for x in ColorKeyModel.numeric_label_spacings])),
            ('pos', Float2Arg),
            ('show_tool', BoolArg),
            ('size', TupleOf(NonNegativeFloatArg,2)),
            ('ticks', BoolArg),
            ('tick_length', NonNegativeFloatArg),
            ('tick_thickness', NonNegativeFloatArg),
        ],
        synopsis = 'Create/change a color key')
    register('key', cmd_desc, key_cmd, logger=logger)

    delete_desc = CmdDesc(synopsis = 'Delete the color key')
    register('key delete', delete_desc, key_delete_cmd, logger=logger)
    create_alias('~key', 'key delete', logger=logger)
    create_alias('key listfonts', '2dlabels listfonts $*', logger=logger)
