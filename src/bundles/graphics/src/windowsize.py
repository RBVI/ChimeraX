# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2022 Regents of the University of California. All rights reserved.
# The ChimeraX application is provided pursuant to the ChimeraX license
# agreement, which covers academic and commercial uses. For more details, see
# <http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
#
# This particular file is part of the ChimeraX library. You can also
# redistribute and/or modify it under the terms of the GNU Lesser General
# Public License version 2.1 as published by the Free Software Foundation.
# For more details, see
# <https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html>
#
# THIS SOFTWARE IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND, EITHER
# EXPRESSED OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
# OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. ADDITIONAL LIABILITY
# LIMITATIONS ARE DESCRIBED IN THE GNU LESSER GENERAL PUBLIC LICENSE
# VERSION 2.1
#
# This notice must be embedded in or attached to all copies, including partial
# copies, of the software or any revisions or derivations thereof.
# === UCSF ChimeraX Copyright ===

def window_size(session, width=None, height=None):
    '''Report or set graphics window size in pixels.'''

    v = session.main_view
    ui = session.ui
    if ui.is_gui:
        # window_size attr is not accurate if graphics
        # haven't been shown...
        ui.main_window.rapid_access_shown = False
    w, h = v.window_size
    if width is None and height is None:
        msg = 'window size %d %d' % (w, h)
        log = session.logger
        log.status(msg)
        log.info(msg)
    else:
        if width is None:
            width = w
        if height is None:
            height = h
        if not ui.is_gui:
            v.window_size = width, height
        elif width != w or height != h:
            ui.main_window.adjust_size(width-w, height-h)
            ui.processEvents()	# Make sure window resized before executing next command.
