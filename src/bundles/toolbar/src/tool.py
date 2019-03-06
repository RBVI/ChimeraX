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
        "show_button_labels": True,
        "show_group_labels": True,
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
        $.fn.ribbon.switchToTabByIndex(1);
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
        button_labels = QAction("Show button labels", menu)
        button_labels.setCheckable(True)
        button_labels.setChecked(self.settings.show_button_labels)
        button_labels.toggled.connect(lambda arg, f=self._set_button_labels: f(arg))
        menu.addAction(button_labels)
        group_labels = QAction("Show group labels", menu)
        group_labels.setCheckable(True)
        group_labels.setChecked(self.settings.show_group_labels)
        group_labels.toggled.connect(lambda arg, f=self._set_group_labels: f(arg))
        menu.addAction(group_labels)

    def _set_button_labels(self, show_button_labels):
        self.settings.show_button_labels = show_button_labels
        self._build_ui()

    def _set_group_labels(self, show_group_labels):
        self.settings.show_group_labels = show_group_labels
        self._build_ui()

    def handle_scheme(self, url):
        # First check that the path is a real command
        cmd = url.path()
        kind, value = cmd.split(':', maxsplit=1)
        if kind == "shortcut":
            self.session.keyboard_shortcuts.run_shortcut(value)
        elif kind == "mouse":
            button_to_bind = 'right'
            from chimerax.core.commands import run
            run(self.session, f'ui mousemode {button_to_bind} "{value}"')
        elif kind == "cmd":
            from chimerax.core.commands import run
            run(self.session, f'{value}')
        else:
            from chimerax.core.errors import UserError
            raise UserError("unknown toolbar command: %s" % cmd)

    def _build_buttons(self):
        import os
        import chimerax.shortcuts
        from PyQt5.QtCore import QUrl
        show_button_labels = self.settings.show_button_labels
        show_group_labels = self.settings.show_group_labels
        shortcut_icon_dir = os.path.join(chimerax.shortcuts.__path__[0], 'icons')
        dir_path = os.path.dirname(__file__)
        qurl = QUrl.fromLocalFile(dir_path)
        html = _prolog.replace("URLBASE", qurl.url())
        if show_button_labels:
            html += _normal_style
        else:
            html += _compact_style
        html += _end_prolog
        for tab in _Toolbars:
            help_url, info = _Toolbars[tab]
            tab_id = tab.replace(' ', '_')
            html += f'''<div class="ribbon-tab" id="{tab_id}-tab">\n\
<span class="ribbon-title">{tab}</span>\n'''
            for (section, compact) in info:
                shortcuts = info[(section, compact)]
                html += '''  <div class="ribbon-section">\n'''
                if show_group_labels:
                    html += f'''  <span class="section-title">{section}</span>\n'''
                for what, icon_file, descrip, tooltip in shortcuts:
                    kind, value = what.split(':', 1)
                    if kind == "mouse":
                        m = self.session.ui.mouse_modes.named_mode(value)
                        if m is None:
                            continue
                        icon_path = m.icon_path
                    else:
                        icon_path = os.path.join(shortcut_icon_dir, icon_file)
                        if not os.path.exists(icon_path):
                            icon_path = icon_file
                    qurl = QUrl.fromLocalFile(icon_path)
                    icon_path = qurl.url()
                    size = "small" if compact else "large"
                    html += f'''    <div class="ribbon-button ribbon-button-{size}" id="{what}">\n'''
                    if show_button_labels:
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
    "File": (
        None,
        {
            ("", False): [
                ("cmd:open browse", "lib/open-in-app.svg", "Open", "Open data file"),
                ("cmd:save browse", "lib/content-save.svg", "Save", "Save session file"),
                ("cmd:close session", "lib/close-box.svg", "Close", "Close current session"),
                ("cmd:exit", "lib/exit.png", "Exit", "Exit application")],
        },
    ),
    "Home": (
        None,
        {
            ("Atoms", True): [
                ("shortcut:da", "atomshow.png", "Show", "Show atoms"),
                ("shortcut:ha", "atomhide.png", "Hide", "Hide atoms")],
            ("Cartoons", True): [
                ("shortcut:rb", "ribshow.png", "Show", "Show cartoons"),
                ("shortcut:hr", "ribhide.png", "Hide", "Hide cartoons")],
            ("Styles", False): [
                ("shortcut:st", "stick.png", "Stick", "Display atoms in stick style"),
                ("shortcut:sp", "sphere.png", "Sphere", "Display atoms in sphere style"),
                ("shortcut:bs", "ball.png", "Ball-and-stick", "Display atoms in ball and stick style")],
            ("Background", False): [
                ("shortcut:wb", "whitebg.png", "White", "White background"),
                ("shortcut:bk", "blackbg.png", "Black", "Black background")],
            ("Lighting", False): [
                ("shortcut:ls", "simplelight.png", "Simple", "Simple lighting"),
                ("shortcut:la", "softlight.png", "Soft", "Ambient lighting"),
                ("shortcut:lf", "fulllight.png", "Full", "Full lighting")],
        },
    ),
    "Molecule Display": (
        "help:user/tools/moldisplay.html",
        {
            ("Last action", True): [
                ("cmd:undo", "lib/undo-variant.png", "Undo", "Undo last action"),
                ("cmd:redo", "lib/redo-variant.png", "Redo", "Redo last action")],
            ("Atoms", True): [
                ("shortcut:da", "atomshow.png", "Show", "Show atoms"),
                ("shortcut:ha", "atomhide.png", "Hide", "Hide atoms")],
            ("Cartoons", True): [
                ("shortcut:rb", "ribshow.png", "Show", "Show cartoons"),
                ("shortcut:hr", "ribhide.png", "Hide", "Hide cartoons")],
            ("Surfaces", True): [
                ("shortcut:ms", "surfshow.png", "Show", "Show surfaces"),
                ("shortcut:hs", "surfhide.png", "Hide", "Hide surfaces")],
            ("Styles", False): [
                ("shortcut:st", "stick.png", "Stick", "Display atoms in stick style"),
                ("shortcut:sp", "sphere.png", "Sphere", "Display atoms in sphere style"),
                ("shortcut:bs", "ball.png", "Ball-and-stick", "Display atoms in ball and stick style")],
            ("Color Atoms", False): [
                ("shortcut:ce", "colorbyelement.png", "By element", "Color non-carbon atoms by element"),
                ("shortcut:cc", "colorbychain.png", "By chain", "Color chains")],
            ("Misc", False): [
                ("shortcut:hb", "hbonds.png", "Show hydrogen bonds", "Show hydrogen bonds")],
        },
    ),
    "Graphics": (
        "help:user/tools/graphics.html",
        {
            ("Last action", True): [
                ("cmd:undo", "lib/undo-variant.png", "Undo", "Undo last action"),
                ("cmd:redo", "lib/redo-variant.png", "Redo", "Redo last action")],
            ("Background", True): [
                ("shortcut:wb", "whitebg.png", "White", "White background"),
                ("shortcut:gb", "graybg.png", "Gray", "Gray background"),
                ("shortcut:bk", "blackbg.png", "Black", "Black background")],
            ("Lighting", False): [
                ("shortcut:ls", "simplelight.png", "Simple", "Simple lighting"),
                ("shortcut:sh", "shadow.png", "Single<br/>shadow", "Toggle shadows"),
                ("shortcut:la", "softlight.png", "Soft", "Ambient lighting"),
                ("shortcut:lf", "fulllight.png", "Full", "Full lighting"),
                ("shortcut:lF", "flat.png", "Flat", "Flat lighting"),
                ("shortcut:se", "silhouette.png", "Silhouettes", "Toggle silhouettes")],
            ("Camera", False): [
                ("shortcut:va", "viewall.png", "View all", "View all"),
                ("shortcut:dv", "orient.png", "Default<br/>orientation", "Default orientation")],
            ("Images", False): [
                ("shortcut:sx", "camera.png", "Save snapshot<br/>to desktop", "Save snapshot to desktop"),
                ("shortcut:vd", "video.png", "Record<br/>spin movie", "Record spin movie")],
        }
    ),
    "Density Map": (
        "help:user/tools/densitymaps.html",
        {
            ("Map", False): [
                ("shortcut:sM", "showmap.png", "Show", "Show map"),
                ("shortcut:hM", "hidemap.png", "Hide", "Hide map")],
            ("Style", False): [
                ("shortcut:fl", "mapsurf.png", "As surface", "Show map or surface in filled style"),
                ("shortcut:me", "mesh.png", "As mesh", "Show map or surface as mesh"),
                ("shortcut:gs", "mapimage.png", "As image", "Show map as grayscale"),
                ("shortcut:tt", "icecube.png", "Transparent<br/>surface", "Toggle surface transparency"),
                ("shortcut:ob", "outlinebox.png", "Outline<br/>box", "Toggle outline box")],
            ("Steps", False): [
                ("shortcut:s1", "step1.png", "Step 1", "Show map at step 1"),
                ("shortcut:s2", "step2.png", "Step 2", "Show map at step 2")],
            ("Fitting", False): [
                ("shortcut:fT", "fitmap.png", "Fit in map", "Fit map in map"),
                ("shortcut:sb", "diffmap.png", "Compute<br/>difference", "Subtract map from map"),
                ("shortcut:gf", "smooth.png", "Smooth", "Smooth map")],
            ("Planes", False): [
                ("shortcut:pl", "plane.png", "One<br/>plane", "Show one plane"),
                ("shortcut:o3", "orthoplanes.png", "Orthogonal<br/>planes", "Show 3 orthogonal planes"),
                ("shortcut:pa", "fullvolume.png", "Full<br/>volume", "Show all planes"),
                ("shortcut:zs", "xyzslice.png", "xyz<br/>slices", "Volume xyz slices"),
                ("shortcut:ps", "perpslice.png", "Perpendicular<br/>slices", "Volume perpendicular slices")],
            ("Misc", False): [
                ("shortcut:aw", "airways.png", "Airways<br/>preset", "Airways preset"),
                ("shortcut:dc", "initialcurve.png", "Default<br/>curve", "Default volume curve")],
        }
    ),
    "Right Mouse": (
        "help:user/tools/mousemodes.html",
        {
            ("Models", False): [
                ("mouse:select", None, "Select", "Select models"),
                ("mouse:rotate", None, "Rotate", "Rotate models"),
                ("mouse:translate", None, "Translate", "Translate models"),
                ("mouse:zoom", None, "Zoom", "Zoom view"),
                ("mouse:rotate and select", None, "Select and<br/>Rotate", "Select and rotate models"),
                ("mouse:translate selected", None, "Translate<br/>Selected", "Translate selected models"),
                ("mouse:rotate selected", None, "Rotate<br/>Selected", "Rotate selected models")],
            ("Clip", False): [
                ("mouse:clip", None, "Clip", "Activate<br/>clipping"),
                ("mouse:clip rotate", None, "Rotate<br/>Clipping", "Rotate clipping planes"),
                ("mouse:zone", None, "Display<br/>zone", "Limit display to zone around clicked residues")],
            ("Label", False): [
                ("mouse:label", None, "Toggle atom or<br/>cartoon label", None),
                ("mouse:move label", None, "Move 2D<br/>label", "Reposition 2D label")],
            ("Misc", False): [
                ("mouse:pivot", None, "Set COFR", "Set center of rotation at atom"),
                ("mouse:bond rotation", None, "Adjust<br/>torsion", "Adject torsion angle"),
                ("mouse:distance", None, "Toggle<br/>distance", "Toggle distance monitor between two atoms"),
                ("mouse:swappaa", None, "Mutate residue", "Mutate and label residue")],
            ("Dynamics", False): [
                ("mouse:tug", None, "Tug atom", "Drag atom while applying dynamics"),
                ("mouse:minimize", None, "Jiggle<br/>residue", "Jiggle residue and its neighbors")],
            ("Volumes", False): [
                ("mouse:place marker", None, "Place<br/>marker", None),
                ("mouse:contour level", None, "Adjust<br/>threshold", "Adjust volume data threshold level"),
                ("mouse:windowing", None, "Adjust<br/>collectively", "Adjust volume data thresholds collectively"),
                ("mouse:move planes", None, "Move planes", "Move plane or slab along its axis to show a different section"),
                ("mouse:crop volume", None, "Crop", "Crop volume data dragging any face of box outline"),
                ("mouse:play map series", None, "Play<br/>series", "Play volume series")],
        }
    ),
}
