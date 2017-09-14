# vim: set expandtab shiftwidth=4 softtabstop=4:

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

def palette_list(session, which="all"):
    '''
    List available palette names.
    '''
    if which == "all" or which == "custom":
        _list(session, session.user_colormaps, "custom")
    if which == "all" or which == "builtin":
        from .. import colors
        _list(session, colors.BuiltinColormaps, "builtin",
              "http://rbvi.ucsf.edu/chimerax/docs/user/commands/palettes.html")

def _list(session, colormaps, kind, url=None):
    from . import cli
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
    noun = cli.plural_form(palettes, "palette")
    if url is None or not is_html:
        label = "%d %s %s: " % (len(palettes), kind, noun)
    else:
        label = "%d <a href=\"%s\">%s %s</a>: " % (len(palettes), url, kind, noun)
    logger.info(label + cli.commas(palettes, " and") + '.', is_html=is_html)

# -----------------------------------------------------------------------------
#
def register_command(session):
    from . import register, CmdDesc, EnumOf
    desc = CmdDesc(optional=[("which", EnumOf(["all", "custom", "builtin"]))],
                   synopsis="list palettes")
    register('palette list', desc, palette_list, logger=session.logger)
