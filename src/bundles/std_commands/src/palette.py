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

def palette_list(session, which="all"):
    '''
    List available palette names.
    '''
    if which == "all" or which == "custom":
        _list(session, session.user_colormaps, "custom")
    if which == "all" or which == "builtin":
        from chimerax.core import colors
        _list(session, colors.BuiltinColormaps, "builtin",
              "http://rbvi.ucsf.edu/chimerax/docs/user/commands/palettes.html")

def _list(session, colormaps, kind, url=None):
    from chimerax.core.commands import plural_form, commas
    import html
    logger = session.logger
    if not colormaps:
        logger.info("No %s palettes." % kind)
        return
    is_html = session.ui.is_gui
    palettes = []
    for name, cm in colormaps.items():
        if cm.name is not None:
            name = cm.name
        if is_html:
            palettes.append(html.escape(name))
        else:
            palettes.append(name)
    noun = plural_form(palettes, "palette")
    if url is None or not is_html:
        label = "%d %s %s: " % (len(palettes), kind, noun)
    else:
        label = "%d <a href=\"%s\">%s %s</a>: " % (len(palettes), url, kind, noun)
    logger.info(label + commas(palettes, "and") + '.', is_html=is_html)

# -----------------------------------------------------------------------------
#
def register_command(logger):
    from chimerax.core.commands import register, CmdDesc, EnumOf
    desc = CmdDesc(optional=[("which", EnumOf(["all", "custom", "builtin"]))],
                   synopsis="list palettes")
    register('palette list', desc, palette_list, logger=logger)
