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

# -----------------------------------------------------------------------------
# User interface for building cages.
#
from chimerax.core.tools import ToolInstance

# ------------------------------------------------------------------------------
#
class Scalebar(ToolInstance):
    help = "help:user/tools/scalebar.html"

    def __init__(self, session, tool_name):
        ToolInstance.__init__(self, session, tool_name)

        from chimerax.ui import MainToolWindow
        tw = MainToolWindow(self, close_destroys=False)
        self.tool_window = tw
        parent = tw.ui_area

        from chimerax.ui.widgets import vertical_layout, EntriesRow, ColorButton
        vertical_layout(parent, margins = (5,0,0,0))

        cur_length, cur_color, cur_xpos, cur_ypos, cur_thickness =  self._scalebar_settings()
        l1 = EntriesRow(parent, self._scalebar_shown, 'Show scalebar.',
                        'Length', cur_length, 'Color', ColorButton,
                        'X', cur_xpos, 'Y', cur_ypos, 'Thickness', cur_thickness, '(pixels)')
        self._show, self._length, self._color, self._xpos, self._ypos, self._thickness = \
            show, length, color, xpos, ypos, h = l1.values
        show.changed.connect(self._update_scalebar)
        length.widget.returnPressed.connect(self._update_scalebar)
        color.color_changed.connect(self._color_changed)
        color.color = 'white' if cur_color is None else cur_color
        self._user_set_color = (cur_color is not None)
        xpos.widget.returnPressed.connect(self._update_scalebar)
        ypos.widget.returnPressed.connect(self._update_scalebar)
        h.widget.returnPressed.connect(self._update_scalebar)
        for e in (length, xpos, ypos, h):
            e.widget.setMaximumWidth(30)

        tw.manage(placement="side")

    @classmethod
    def get_singleton(self, session, create=True):
        from chimerax.core import tools
        return tools.get_singleton(session, Scalebar, 'Scale Bar', create=create)

    def _color_changed(self):
        self._user_set_color = True
        self._update_scalebar()
        
    def _update_scalebar(self):
        if self._show.enabled:
            self._show_scalebar()
        else:
            self._close_scalebar()
            
    def _show_scalebar(self):
        options = self._changed_options()
        if options or not self._scalebar_shown:
            cmd = 'scalebar ' + ' '.join(options)
            from chimerax.core.commands import run
            run(self.session, cmd)
        
    def _close_scalebar(self):
        if self._scalebar_shown:
            cmd = 'scalebar off'
            from chimerax.core.commands import run
            run(self.session, cmd)

    @property
    def _scalebar_shown(self):
        from .scalebar import _scalebar_label
        return _scalebar_label(self.session) is not None
        
    def _changed_options(self):
        cur_length, cur_color, cur_xpos, cur_ypos, cur_thickness =  self._scalebar_settings()

        length = self._length.value
        color = self._color.color	# rgba8
        xpos, ypos = self._xpos.value, self._ypos.value
        thickness = self._thickness.value

        options = []
        if length != cur_length:
            options.append('%.5g' % length)
        set_color = self._user_set_color if cur_color is None else tuple(color) != tuple(cur_color)
        if set_color:
            from chimerax.core.colors import hex_color
            options.append('color %s' % hex_color(color))
        if xpos != cur_xpos:
            options.append('xpos %.5g' % xpos)
        if ypos != cur_ypos:
            options.append('ypos %.5g' % ypos)
        if thickness != cur_thickness:
            options.append('thickness %.5g' % thickness)
                
        return options

    def _scalebar_settings(self):
        from .scalebar import _scalebar_label
        s = _scalebar_label(self.session)
        if s:
            return s.scalebar_width, s.color, s.xpos, s.ypos, s.scalebar_height
        return 100, None, 0.1, 0.1, 10
