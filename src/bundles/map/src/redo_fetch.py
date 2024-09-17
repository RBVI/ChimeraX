# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2016 Regents of the University of California.
# All rights reserved.  This software provided pursuant to a
# license agreement containing restrictions on its disclosure,
# duplication and use.  For details see:
# https://www.rbvi.ucsf.edu/chimerax/docs/licensing.html
# This notice must be embedded in or attached to all copies,
# including partial copies, of the software or any revisions
# or derivations thereof.
# === UCSF ChimeraX Copyright ===

# -----------------------------------------------------------------------------
#
def fetch_mtz_map(session, pdb_id, ignore_cache=False, **kw):
    """Get map file from PDB-REDO repository"""
    from chimerax.core.errors import UserError
    if len(pdb_id) not in (4,8):
        raise UserError('PDB identifiers are either 4 or 8 characters long, got "%s"' % pdb_id)

    from chimerax.open_command import NoOpenerError
    try:
        fmt = session.data_formats.open_format_from_suffix(".mtz")
    except NoOpenerError:
        raise UserError("Don't know how to open .mtz map files")
    bundle_info = session.open_command.provider_info(fmt).bundle_info
    if not bundle_info.installed:
        # Issuing the log message immediately here causes it to appear in the table
        # that summarizes file-opening messages, and we'd like to avoid that
        from chimerax.core import toolshed
        ts = toolshed.get_toolshed()
        msg = '<a href="%s">Install the %s bundle</a> from the Toolshed.' % (
            ts.bundle_url(bundle_info.name), bundle_info.short_name)
        if session.ui.is_gui:
            from Qt.QtCore import QTimer
            QTimer.singleShot(0, lambda *args, session=session, msg=msg:
                session.logger.info(msg, is_html=True))
        else:
            session.logger.info(msg, is_html=True)
        raise UserError("You need to install the %s bundle to open PDB-REDO MTZ-format map files."
            "  See the log for details." % bundle_info.short_name)

    from chimerax.mmcif.mmcif import pdb_redo_base_url
    pdb_id, base_url = pdb_redo_base_url(pdb_id)
    from chimerax.core.fetch import fetch_file
    map_name = "%s.mtz" % pdb_id
    filename = fetch_file(session, base_url + ".mtz", 'MTZ %s' % pdb_id, map_name,
        "PDB-REDO", ignore_cache=ignore_cache)
    session.logger.status("Opening PDB-REDO map %s" % (pdb_id,))
    models, status = session.open_command.open_data(filename, format='mtz', name=pdb_id, **kw)
    return models, status
