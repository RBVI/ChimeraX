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

_prolog = """
<html>
  <!-- vi:set expandtab shiftwidth=2: -->
  <head>
    <link href="lib/ribbon/ribbon.css" rel="stylesheet" type="text/css" />
    <script type="text/javascript" src="lib/jquery-1.6.1.min.js"></script>
    <script type="text/javascript" src="lib/ribbon/ribbon.js"></script>
    <script type="text/javascript" src="lib/ribbon/jquery.tooltip.min.js"></script>
    <script type="text/javascript">
      $(document).ready(function () {
	$('#ribbon').ribbon();
      });
    </script>

  </head>
  <body bgcolor="#c9cdd2">
    <div id="ribbon">
      <div class="ribbon-window-title"></div>
"""
_epilogue = """
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
        self._build_ui()

    def _build_ui(self):
        # Fill in html viewer with initial page in the module
        import os.path
        html_file = os.path.join(os.path.dirname(__file__), "tool.html")
        import pathlib
        self.html_view.setUrl(pathlib.Path(html_file).as_uri())

    def handle_scheme(self, url):
        # ``url`` - ``PyQt5.QtCore.QUrl`` instance

        # First check that the path is a real command
        command = url.path()
        if command == "invoke":
            # Collect the optional parameters from URL query parameters
            # and construct a command to execute
            from urllib.parse import parse_qs
            query = parse_qs(url.query())

            # First the command
            cmd_text = ["toolbar", command]

            # TODO: 
        else:
            from chimerax.core.errors import UserError
            raise UserError("unknown toolbar command: %s" % command)

    def build_buttons(self):
        pass
