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
from chimerax.core.settings import Settings


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
            var cmd = $(this).attr('id');
            var link = document.createElement('a');
            link.href = "toolbar:" + cmd;
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
      .ribbon-tooltip
      {
        width: auto;
        top: auto;
      }
"""
_normal_style = """
      #ribbon .ribbon-button-large .ribbon-icon
      {
        width: 24px;
        height: 24px;
      }
"""
_compact_style = """
      #ribbon .ribbon-button-large
      {
        height: 48px;
      }
      #ribbon .ribbon-button-large .ribbon-icon
      {
        width: 48px;
        height: 48px;
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

    SESSION_ENDURING = True
    SESSION_SAVE = False        # No session saving for now
    PLACEMENT = "top"
    CUSTOM_SCHEME = "toolbar"
    help = "help:user/tools/Toolbar.html"  # Let ChimeraX know about our help page

    def __init__(self, session, tool_name):
        super().__init__(session, tool_name, size_hint=(575, 110), log_errors=True)
        self.display_name = "Toolbar"
        self.settings = ToolbarSettings(session, tool_name)
        self._build_ui()
        self.tool_window.fill_context_menu = self.fill_context_menu
        # kludge to hide title bar
        from PyQt5.QtWidgets import QWidget
        self.tool_window._kludge.dock_widget.setTitleBarWidget(QWidget())
        from chimerax.core.commands import run
        # run(session, "toolshed hide 'Density Map Toolbar'", log=False)
        run(session, "toolshed hide 'Graphics Toolbar'", log=False)
        run(session, "toolshed hide 'Molecule Display Toolbar'", log=False)
        run(session, "toolshed hide 'Mouse Modes for Right Button'", log=False)

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
        cmd = url.path()
        kind, value = cmd.split('-', maxsplit=1)
        if kind == "shortcut":
            self.session.keyboard_shortcuts.run_shortcut(value)
        elif kind == "mouse":
            button_to_bind = 'right'
            from chimerax.core.commands import run
            run(self.session, f'ui mousemode {button_to_bind} "{value}"')
        elif kind in ("undo", "redo"):
            from chimerax.core.commands import run
            run(self.session, f'{kind}')
        else:
            from chimerax.core.errors import UserError
            raise UserError("unknown toolbar command: %s" % cmd)

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
            html += _compact_style
        html += _end_prolog
        for tab in _Toolbars:
            help_url, info = _Toolbars[tab]
            tab_id = tab.replace(' ', '_')
            html += f'''<div class="ribbon-tab" id="{tab_id}-tab">\n\
<span class="ribbon-title">{tab}</span>\n'''
            if tab != "Right Mouse":
                html += '''  <div class="ribbon-section">\n'''
                if show_hints:
                    html += '''  <span class="section-title">Last action</span>\n'''
                for action in ("undo", "redo"):
                    html += f'''    <div class="ribbon-button ribbon-button-small" id="{action}-">\n'''
                    title = action.title()
                    if show_hints:
                        html += f'''        <span class="button-title">{title}</span>\n'''
                    html += f'''        <span class="button-help">{title} last action</span>\n'''
                    icon_path = f'lib/{action}-variant.svg'
                    html += f'''        <img class="ribbon-icon ribbon-normal" src="{icon_path}"/>\n\
        </div>\n'''
                html += "  </div>\n"
            for (section, compact) in info:
                shortcuts = info[(section, compact)]
                html += '''  <div class="ribbon-section">\n'''
                if show_hints:
                    html += f'''  <span class="section-title">{section}</span>\n'''
                for what, icon_file, descrip, tooltip in shortcuts:
                    if tab == "Right Mouse":
                        m = self.session.ui.mouse_modes.named_mode(what)
                        if m is None:
                            continue
                        icon_path = m.icon_path
                        cmd_id = f"mouse-{what}"
                    else:
                        icon_path = os.path.join(icon_dir, icon_file)
                        cmd_id = f"shortcut-{what}"
                    qurl = QUrl.fromLocalFile(icon_path)
                    icon_path = qurl.url()
                    size = "small" if compact else "large"
                    html += f'''    <div class="ribbon-button ribbon-button-{size}" id="{cmd_id}">\n'''
                    if show_hints:
                        html += f'''        <span class="button-title">{descrip}</span>\n'''
                    if not tooltip:
                        tooltip = descrip
                    if tooltip:
                        html += f'''        <span class="button-help">{tooltip}</span>\n'''
                    html += f'''        <img class="ribbon-icon ribbon-normal" src="{icon_path}"/>\n\
    </div>\n'''
                html += "  </div>\n"
            html += "</div>\n"
        html += _epilog
        return html


_Toolbars = {
    'Molecule Display': (
        "help:user/tools/moldisplay.html",
        {
            ("Atoms", True): [
                ('da', 'atomshow.png', 'Show', 'Show atoms'),
                ('ha', 'atomhide.png', 'Hide', 'Hide atoms')],
            ("Cartoons", True): [
                ('rb', 'ribshow.png', 'Show', 'Show cartoons'),
                ('hr', 'ribhide.png', 'Hide', 'Hide cartoons')],
            ("Surfaces", True): [
                ('ms', 'surfshow.png', 'Show', 'Show surfaces'),
                ('hs', 'surfhide.png', 'Hide', 'Hide surfaces')],
            ("Styles", False): [
                ('st', 'stick.png', 'Stick', 'Display atoms in stick style'),
                ('sp', 'sphere.png', 'Sphere', 'Display atoms in sphere style'),
                ('bs', 'ball.png', 'Ball-and-stick', 'Display atoms in ball and stick style')],
            ("Color Atoms", False): [
                ('ce', 'colorbyelement.png', 'By element', 'Color non-carbon atoms by element'),
                ('cc', 'colorbychain.png', 'By chain', 'Color chains')],
            ("Misc", False): [
                ('hb', 'hbonds.png', 'Show hydrogen bonds', 'Show hydrogen bonds')],
        },
    ),
    "Graphics": (
        "help:user/tools/graphics.html",
        {
            ("Background", True): [
                ('wb', 'whitebg.png', 'White', 'White background'),
                ('gb', 'graybg.png', 'Gray', 'Gray background'),
                ('bk', 'blackbg.png', 'Black', 'Black background')],
            ("Lighting", False): [
                ('ls', 'simplelight.png', 'Simple', 'Simple lighting'),
                ('sh', 'shadow.png', 'Single<br/>shadow', 'Toggle shadows'),
                ('la', 'softlight.png', 'Soft', 'Ambient lighting'),
                ('lf', 'fulllight.png', 'Full', 'Full lighting'),
                ('lF', 'flat.png', 'Flat', 'Flat lighting'),
                ('se', 'silhouette.png', 'Silhouettes', 'Toggle silhouettes')],
            ("Camera", False): [
                ('va', 'viewall.png', 'View all', 'View all'),
                ('dv', 'orient.png', 'Default<br/>orientation', 'Default orientation')],
            ("Images", False): [
                ('sx', 'camera.png', 'Save snapshot<br/>to desktop', 'Save snapshot to desktop'),
                ('vd', 'video.png', 'Record<br/>spin movie', 'Record spin movie')],
        }
    ),
    "Density Map": (
        "help:user/tools/densitymaps.html",
        {
            ("Map", False): [
                ('sM', 'showmap.png', 'Show', 'Show map'),
                ('hM', 'hidemap.png', 'Hide', 'Hide map')],
            ("Style", False): [
                ('fl', 'mapsurf.png', 'As surface', 'Show map or surface in filled style'),
                ('me', 'mesh.png', 'As mesh', 'Show map or surface as mesh'),
                ('gs', 'mapimage.png', 'As image', 'Show map as grayscale'),
                ('tt', 'icecube.png', 'Transparent<br/>surface', 'Toggle surface transparency'),
                ('ob', 'outlinebox.png', 'Outline<br/>box', 'Toggle outline box')],
            ("Steps", False): [
                ('s1', 'step1.png', 'Step 1', 'Show map at step 1'),
                ('s2', 'step2.png', 'Step 2', 'Show map at step 2')],
            ("Fitting", False): [
                ('fT', 'fitmap.png', 'Fit in map', 'Fit map in map'),
                ('sb', 'diffmap.png', 'Compute<br/>difference', 'Subtract map from map'),
                ('gf', 'smooth.png', 'Smooth', 'Smooth map')],
            ("Planes", False): [
                ('pl', 'plane.png', 'One<br/>plane', 'Show one plane'),
                ('o3', 'orthoplanes.png', 'Orthogonal<br/>planes', 'Show 3 orthogonal planes'),
                ('pa', 'fullvolume.png', 'Full<br/>volume', 'Show all planes'),
                ('zs', 'xyzslice.png', 'xyz<br/>slices', 'Volume xyz slices'),
                ('ps', 'perpslice.png', 'Perpendicular<br/>slices', 'Volume perpendicular slices')],
            ("Misc", False): [
                ('aw', 'airways.png', 'Airways<br/>preset', 'Airways preset'),
                ('dc', 'initialcurve.png', 'Default<br/>curve', 'Default volume curve')],
        }
    ),
    "Right Mouse": (
        "help:user/tools/mousemodes.html",
        {
            ("Models", False): [
                ('select', None, 'Select', 'Select models'),
                ('rotate', None, 'Rotate', 'Rotate models'),
                ('translate', None, 'Translate', 'Translate models'),
                ('zoom', None, 'Zoom', 'Zoom view'),
                ('rotate and select', None, 'Select and<br/>Rotate', 'Select and rotate models'),
                ('translate selected', None, 'Translate<br/>Selected', 'Translate selected models'),
                ('rotate selected', None, 'Rotate<br/>Selected', 'Rotate selected models')],
            ("Clip", False): [
                ('clip', None, 'Clip', 'Activate<br/>clipping'),
                ('clip rotate', None, 'Rotate<br/>Clipping', 'Rotate clipping planes'),
                ('zone', None, 'Display<br/>zone', 'Limit display to zone around clicked residues')],
            ("Label", False): [
                ('label', None, 'Toggle atom or<br/>cartoon label', None),
                ('move label', None, 'Move 2D<br/>label', 'Reposition 2D label')],
            ("Misc", False): [
                ('pivot', None, 'Set COFR', 'Set center of rotation at atom'),
                ('bond rotation', None, 'Adjust<br/>torsion', 'Adject torsion angle'),
                ('distance', None, 'Toggle<br/>distance', 'Toggle distance monitor between two atoms'),
                ('swappaa', None, 'Mutate residue', 'Mutate and label residue')],
            ("Dynamics", False): [
                ('tug', None, 'Tug atom', 'Drag atom while applying dynamics'),
                ('minimize', None, 'Jiggle<br/>residue', 'Jiggle residue and its neighbors')],
            ("Volumes", False): [
                ('tug', None, 'Tug atom', 'Drag atom while applying dynamics'),
                ('place marker', None, 'Place<br/>marker', None),
                ('contour level', None, 'Adjust<br/>threshold', 'Adjust volume data threshold level'),
                ('windowing', None, 'Adjust<br/>collectively', 'Adjust volume data thresholds collectively'),
                ('move planes', None, 'Move planes', 'Move plane or slab along its axis to show a different section'),
                ('crop volume', None, 'Crop', 'Crop volume data dragging any face of box outline'),
                ('play map series', None, 'Play<br/>series', 'Play volume series')],
        }
    ),
}
