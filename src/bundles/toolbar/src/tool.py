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
            //alert(shortcut + ' clicked');
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
        height: 120px;
      }
      #ribbon .ribbon-tab
      {
        padding: 0px;
      }
      #ribbon .ribbon-button-large .ribbon-icon
      {
        width: 32px;
        height: 32px;
      }
    </style>

  </head>
  <body bgcolor="#c9cdd2">
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
    CUSTOM_SCHEME = "toolbar"   # Scheme used in HTML for callback into Python
    help = "help:user/tools/Toolbar.html"  # Let ChimeraX know about our help page

    def __init__(self, session, tool_name):
        super().__init__(session, tool_name, size_hint=(575, 215), log_errors=True)
        self.display_name = "Toolbar"
        from chimerax.shortcuts import shortcuts
        self.keyboard_shortcuts = shortcuts.keyboard_shortcuts(session)
        self._build_ui()

    def _build_ui(self):
        from PyQt5.QtCore import QUrl
        html = self.build_buttons()
        #with open('debug.html', 'w') as f:
        #    f.write(html)
        self.html_view.setHtml(html, QUrl("file://"))

    def handle_scheme(self, url):
        # First check that the path is a real command
        keys = url.path()
        if len(keys) != 2:
            from chimerax.core.errors import UserError
            raise UserError("unknown toolbar command: %s" % keys)
        self.keyboard_shortcuts.run_shortcut(keys)

    def build_buttons(self):
        import os
        import chimerax.shortcuts
        from PyQt5.QtCore import QUrl
        icon_dir = os.path.join(chimerax.shortcuts.__path__[0], 'icons')
        dir_path = os.path.dirname(__file__)
        qurl = QUrl.fromLocalFile(dir_path)
        html = _prolog.replace("URLBASE", qurl.url())
        for tab in _Toolbars:
            help_url, info = _Toolbars[tab]
            html += f'''<div class="ribbon-tab" id="{tab.replace(' ', '_')}-tab">\n\
<span class="ribbon-title">{tab}</span>\n'''
            for section in info:
                shortcuts = info[section]
                html += f'''  <div class="ribbon-section">\n\
  <span class="section-title">{section}</span>\n'''
                for keys, icon_file, descrip, small in shortcuts:
                    qurl = QUrl.fromLocalFile(os.path.join(icon_dir, icon_file))
                    icon_path = qurl.url()
                    size = "small" if small else "large"
                    html += f'''    <div class="ribbon-button ribbon-button-{size}" id="{keys}-btn">\n\
        <span class="button-title">{descrip}</span>\n\
        <img class="ribbon-icon ribbon-normal" src="{icon_path}"/>\n\
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
