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

import re
ColorNames = re.compile(r'[a-z][-_a-z0-9 ]*')


def html_color_swatch(color):
    return (
        '&nbsp;<div style="width:1em; height:.6em; display:inline-block;'
        ' border:1px solid #000; background-color:%s"></div>'
        % color.hex())


def name_color(session, name, color):
    """Create a custom color."""
    if ColorNames.match(name) is None:
        from chimerax.core.errors import UserError
        raise UserError('Illegal color name: "%s"' % name)

    name = ' '.join(name.split())   # canonicalize
    session.user_colors.add(name, color)
    show_color(session, name)


def show_color(session, name):
    """Show color in log."""
    if ColorNames.match(name) is None:
        from chimerax.core.errors import UserError
        raise UserError('Illegal color name: "%s"' % name)

    name = ' '.join(name.split())   # canonicalize

    from chimerax.core.commands.colorarg import find_named_color
    if session is not None:
        real_name, color, rest = find_named_color(session.user_colors, name)
        if rest:
            color = None
    else:
        from chimerax.core.colors import BuiltinColors
        real_name, color, rest = find_named_color(BuiltinColors, name)
        if rest:
            color = None
    if color is None:
        from chimerax.core.errors import UserError
        raise UserError('Unknown color %r' % name)

    def percent(x):
        if x == 1:
            return 100
        return ((x * 10000) % 10000) / 100
    red, green, blue, alpha = color.rgba
    if alpha >= 1:
        if red == green == blue:
            text = "gray(%.3g%%) hex: %s" % (percent(red), color.hex())
        else:
            text = "rgb(%.3g%%, %.3g%%, %.3g%%) hex: %s" % (
                percent(red), percent(green), percent(blue), color.hex())
    else:
        text = "rgba(%.3g%%, %.3g%%, %.3g%%, %.3g%%) hex: %s" % (
            percent(red), percent(green), percent(blue), percent(alpha), color.hex_with_alpha())

    if alpha >= 1:
        transmit = 'opaque'
    elif alpha <= 0:
        transmit = '100% transparent'
    else:
        transmit = '%.4g%% transparent' % percent(1 - alpha)

    msg = 'Color %r is %s: %s' % (real_name, transmit, text)
    if session is None:
        print(msg)
        return
    if not session.ui.is_gui:
        session.logger.info(msg)
    else:
        session.logger.status(msg)
        session.logger.info(
            msg + html_color_swatch(color), is_html=True)


def delete_color(session, name):
    """Remove a custom color."""
    if name == 'custom':
        color_names = session.user_colors.list()
        for name in color_names:
            session.user_colors.remove(name)
        return
    if name not in session.user_colors:
        from chimerax.core.errors import UserError
        raise UserError('Unknown color %r' % name)
    try:
        session.user_colors.remove(name)
    except ValueError as v:
        from chimerax.core.errors import UserError
        raise UserError(v)


def list_colors(session, which='all'):
    if which == 'all' or which == 'custom':
        from sortedcontainers import SortedDict
        d = SortedDict([(name, session.user_colors[name])
                        for name in session.user_colors.list()])
        _list_colors(session, d, 'custom')
    if which == 'all' or which == 'builtin':
        from chimerax.core import colors
        _list_colors(session, colors.BuiltinColors, 'builtin',
                     "http://rbvi.ucsf.edu/chimerax/docs/user/commands/"
                     "colornames.html#builtin")


def _list_colors(session, colors_dict, kind, url=None):
    from chimerax.core.commands import commas, plural_form
    import html
    logger = session.logger
    if not colors_dict:
        logger.info("No %s colors." % kind)
        return
    is_html = session.ui.is_gui
    colors = []
    for name, c in colors_dict.items():
        if is_html:
            colors.append(html.escape(name) + html_color_swatch(c))
        else:
            colors.append(name)
    noun = plural_form(colors, 'color')
    if url is None or not is_html:
        label = "%d %s %s: " % (len(colors), kind, noun)
    else:
        label = "%d <a href=\"%s\">%s %s</a>: " % (len(colors), url, kind, noun)
    logger.info(label + commas(colors, 'and') + '.', is_html=is_html)


# -----------------------------------------------------------------------------
#
def register_command(logger):
    from chimerax.core.commands import register, CmdDesc, StringArg, ColorArg, NoArg, EnumOf
    from chimerax.core.commands import Or, RestOfLine, create_alias
    register(
        'color list',
        CmdDesc(
            optional=[('which', EnumOf(["all", "custom", "builtin"]))],
            synopsis='list colors'),
        list_colors, logger=logger
    )

    register(
        'color show',
        CmdDesc(required=[('name', RestOfLine)],
                synopsis="show color"),
        show_color, logger=logger
    )
    register(
        'color name',
        CmdDesc(required=[('name', StringArg), ('color', ColorArg)],
                synopsis="name a custom color"),
        name_color, logger=logger
    )
    register(
        'color delete',
        CmdDesc(required=[('name', Or(EnumOf(['custom']), StringArg))],
                synopsis="remove color definition"),
        delete_color, logger=logger
    )
    create_alias('colordef', 'color name $*', logger=logger)
    create_alias('~colordef', 'color delete $*', logger=logger)
    create_alias('colourdef', 'color name $*', logger=logger,
            url="help:user/commands/colordef.html")
    create_alias('~colourdef', 'color delete $*', logger=logger,
            url="help:user/commands/colordef.html")
