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

from chimerax.ui import HtmlToolInstance
from  chimerax.core.settings import Settings

class ToolbarSettings(Settings):
    AUTO_SAVE = {
        "show_hints": True,
    }

_prolog = """<html>
  <!-- vi:set expandtab shiftwidth=2: -->
  <head>
    <meta charset="UTF-8">
    <base href="URLBASE/"/>
    <link href="lib/ribbon/ribbon.css" rel="stylesheet" type="text/css"/>
    <script type="text/javascript" src="lib/jquery-1.6.1.min.js"></script>
    <script type="text/javascript" src="lib/ribbon/ribbon.js"></script>
    <script type="text/javascript" src="lib/ribbon/jquery.tooltip.min.js"></script>
    <script type="text/javascript">
      $(document).ready(function () {
        $('#ribbon').ribbon();
        $('.ribbon-button').click(function() {
          if (this.isEnabled()) {
            shortcut = $(this).attr('id').slice(0, 2);
            var link = document.createElement('a');
            link.href = "toolbar:" + shortcut;
            link.click();
          }
        });
      });
    </script>
    <style>
      // tighten up ribbon display
      body {
        padding: 0px;
      }
      #ribbon {
        padding: 0px;
        height: 110px;
        background-color: #f0f0f0;
      }
      #ribbon .ribbon-tab
      {
        padding: 0px;
      }
"""
_normal_style = """
      #ribbon .ribbon-button-large .ribbon-icon
      {
        width: 24px;
        height: 24px;
      }
"""
_small_style = """
      #ribbon .ribbon-button-large
      {
        height: 60px;
      }
      #ribbon .ribbon-button-large .ribbon-icon
      {
        width: 60px;
        height: 60px;
      }
"""
_end_prolog = """
    </style>

  </head>
  <body>
    <div id="ribbon">
      <div class="ribbon-window-title"></div>
"""
_epilog = """
      </div>
    </div>
  </body>
</html>
"""


class ToolbarTool(HtmlToolInstance):

    SESSION_ENDURING = False    # Does this instance persist when session closes
    SESSION_SAVE = False        # No session saving for now
    PLACEMENT = "top"
    CUSTOM_SCHEME = "toolbar"
    help = "help:user/tools/Toolbar.html"  # Let ChimeraX know about our help page

    def __init__(self, session, tool_name):
        super().__init__(session, tool_name, size_hint=(575, 110), log_errors=True)
        self.display_name = "Toolbar"
        self.settings = ToolbarSettings(session, tool_name)
        from chimerax.shortcuts import shortcuts
        self.keyboard_shortcuts = shortcuts.keyboard_shortcuts(session)
        self._build_ui()
        self.tool_window.fill_context_menu = self.fill_context_menu
        # kludge to hide title bar
        from PyQt5.QtWidgets import QWidget
        self.tool_window._kludge.dock_widget.setTitleBarWidget(QWidget())
        from chimerax.core.commands import run
        run(session, "toolshed hide 'Graphics Toolbar'", log=False)
        run(session, "toolshed hide 'Molecule Display Toolbar'", log=False)
        # run(session, "toolshed hide 'Mouse Modes for Right Button'", log=False)

    def _build_ui(self):
        from PyQt5.QtCore import QUrl
        html = self._build_buttons()
        # with open('debug.html', 'w') as f:
        #     f.write(html)
        self.html_view.setHtml(html, QUrl("file://"))

    def fill_context_menu(self, menu, x, y):
        # avoid having actions destroyed when this routine returns
        # by stowing a reference in the menu itself
        from PyQt5.QtWidgets import QAction
        hint_action = QAction("Show hints", menu)
        hint_action.setCheckable(True)
        hint_action.setChecked(self.settings.show_hints)
        hint_action.toggled.connect(lambda arg, f=self._set_show_hints: f(arg))
        menu.addAction(hint_action)

    def _set_show_hints(self, show_hints):
        self.settings.show_hints = show_hints
        self._build_ui()

    def handle_scheme(self, url):
        # First check that the path is a real command
        keys = url.path()
        if len(keys) != 2:
            from chimerax.core.errors import UserError
            raise UserError("unknown toolbar command: %s" % keys)
        self.keyboard_shortcuts.run_shortcut(keys)

    def _build_buttons(self):
        import os
        import chimerax.shortcuts
        from PyQt5.QtCore import QUrl
        show_hints = self.settings.show_hints
        icon_dir = os.path.join(chimerax.shortcuts.__path__[0], 'icons')
        dir_path = os.path.dirname(__file__)
        qurl = QUrl.fromLocalFile(dir_path)
        html = _prolog.replace("URLBASE", qurl.url())
        if show_hints:
            html += _normal_style
        else:
            html += _small_style
        html += _end_prolog
        for tab in _Toolbars:
            help_url, info = _Toolbars[tab]
            html += f'''<div class="ribbon-tab" id="{tab.replace(' ', '_')}-tab">\n\
<span class="ribbon-title">{tab}</span>\n'''
            for section in info:
                shortcuts = info[section]
                html += '''  <div class="ribbon-section">\n'''
                if show_hints:
                    html += f'''  <span class="section-title">{section}</span>\n'''
                for keys, icon_file, descrip, small in shortcuts:
                    qurl = QUrl.fromLocalFile(os.path.join(icon_dir, icon_file))
                    icon_path = qurl.url()
                    size = "small" if small else "large"
                    html += f'''    <div class="ribbon-button ribbon-button-{size}" id="{keys}-btn">\n'''
                    if show_hints:
                        html += f'''        <span class="button-title">{descrip}</span>\n'''
                    html += f'''        <img class="ribbon-icon ribbon-normal" src="{icon_path}"/>\n\
    </div>\n'''
                    #<span class="button-help">This button will add a table to your document.</span>
                    #<img class="ribbon-icon ribbon-hot" src="lib/icons/hot/new-table.png" />
                    #<img class="ribbon-icon ribbon-disabled" src="lib/icons/disabled/new-table.png" />
                html += "  </div>\n"
            html += "</div>\n"
        html += _epilog
        return html


_Toolbars = {
    'Molecule Display': (
        "help:user/tools/moldisplay.html",
        {
            "Atoms": [
                ('da', 'atomshow.png', 'Show', True),
                ('ha', 'atomhide.png', 'Hide', True)],
            "Cartoons": [
                ('rb', 'ribshow.png', 'Show', True),
                ('hr', 'ribhide.png', 'Hide', True)],
            "Surfaces": [
                ('ms', 'surfshow.png', 'Show', True),
                ('hs', 'surfhide.png', 'Hide', True)],
            "Styles": [
                ('st', 'stick.png', 'Stick', False),
                ('sp', 'sphere.png', 'Sphere', False),
                ('bs', 'ball.png', 'Ball-and-stick', False)],
            "Color Atoms": [
                ('ce', 'colorbyelement.png', 'By element', False),
                ('cc', 'colorbychain.png', 'By chain', False)],
            "Misc": [
                ('hb', 'hbonds.png', 'Find hydrogen bonds', False)],
        },
    ),
    "Graphics": (
        "help:user/tools/graphics.html",
        {
            "Background": [
                ('wb', 'whitebg.png', 'White', True),
                ('gb', 'graybg.png', 'Gray', True),
                ('bk', 'blackbg.png', 'Black', True)],
            "Lighting": [
                ('ls', 'simplelight.png', 'Simple', False),
                ('sh', 'shadow.png', 'Single<br/>shadow', False),
                ('la', 'softlight.png', 'Soft', False),
                ('lf', 'fulllight.png', 'Full', False),
                ('lF', 'flat.png', 'Flat', False),
                ('se', 'silhouette.png', 'Silhouettes', False)],
            "Camera": [
                ('va', 'viewall.png', 'View all', False),
                ('dv', 'orient.png', 'Standard<br/>orientation', False)],
            "Images": [
                ('sx', 'camera.png', 'Save snapshot<br/>to desktop', False),
                ('vd', 'video.png', 'Record<br/>spin movie', False)],
        }
    ),
    "Density Map": (
        None,
        {
            "Map": [
                ('sM', 'showmap.png', 'Show', False),
                ('hM', 'hidemap.png', 'Hide', False)],
            "Style": [
                ('fl', 'mapsurf.png', 'As surface', False),
                ('me', 'mesh.png', 'As mesh', False),
                ('gs', 'mapimage.png', 'As image', False),
                ('tt', 'icecube.png', 'Transparent<br/>surface', False),
                ('ob', 'outlinebox.png', 'Outline<br/>box', False)],
            "Steps": [
                ('s1', 'step1.png', 'Step 1', False),
                ('s2', 'step2.png', 'Step 2', False)],
            "Fitting": [
                ('fT', 'fitmap.png', 'Fit in map', False),
                ('sb', 'diffmap.png', 'Compute<br/>difference', False),
                ('gf', 'smooth.png', 'Smooth', False)],
            "Planes": [
                ('pl', 'plane.png', 'One<br/>plane', False),
                ('o3', 'orthoplanes.png', 'Orthogonal<br/>planes', False),
                ('pa', 'fullvolume.png', 'Full<br/>volume', False),
                ('zs', 'xyzslice.png', 'xyz<br/>slices', False),
                ('ps', 'perpslice.png', 'Perpendicular<br/>slices', False)],
            "Misc": [
                ('aw', 'airways.png', 'Airways<br/>preset', False),
                ('dc', 'initialcurve.png', 'Default<br/>curve', False)],
        }
    )
}
