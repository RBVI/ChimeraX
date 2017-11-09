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

from chimerax.core.ui import HtmlToolInstance


class TutorialGUI(HtmlToolInstance):

    # Inheriting from HtmlToolInstance gets us the following attributes
    # after initialization:
    #   self.tool_window: instance of chimerax.core.ui.gui.MainToolWindow
    #   self.html_view: instance of chimerax.core.ui.widgets.HtmlView
    # Defining methods in this subclass also trigger some automated callbacks:
    #   handle_scheme: called when custom-scheme link is visited
    #   update_models: called when models are opened or closed
    # If cleaning up is needed on finish, override the ``delete`` method
    # but be sure to call ``delete`` from the superclass at the end.

    SESSION_ENDURING = False    # Does this instance persist when session closes
    SESSION_SAVE = False        # No session saving for now
    CUSTOM_SCHEME = "tutorial"  # Scheme used in HTML for callback into Python

    def __init__(self, session, ti):
        # ``session`` - ``chimerax.core.session.Session`` instance
        # ``ti``      - ``chimerax.core.toolshed.ToolInfo`` instance

        # Set name displayed on title bar
        self.display_name = "Tutorial GUI"

        # Initialize base class.  ``size_hint`` is the suggested
        # initial tool size in pixels.
        super().__init__(session, ti.name, size_hint=(575, 400))
        self._build_ui()

    def _build_ui(self):
        # Fill in html viewer with initial page in the module
        import os.path
        html_file = os.path.join(os.path.dirname(__file__), "gui.html")
        import pathlib
        self.html_view.setUrl(pathlib.Path(html_file).as_uri())

    def handle_scheme(self, url):
        # ``url`` - ``PyQt5.QtCore.QUrl`` instance

        # This method is called the user clicks a link on the HTML page
        # with our custom scheme.  The URL path and query parameters
        # are controlled on the HTML side via Javascript.  Obviously,
        # we still do security checks in case the user somehow was
        # diverted to a malicious page specially crafted with links
        # with our custom scheme.  (Unlikely, but not impossible.)
        # URLs should look like: tutorial:cofm?weighted=1

        # First check that the path is a real command
        command = url.path()
        if command == "update_models":
            self.update_models()
            return
        elif command in ["cofm", "highlight"]:
            # Collect the optional parameters from URL query parameters
            # and construct a command to execute
            from urllib.parse import parse_qs
            query = parse_qs(url.query())

            # First the command
            cmd_text = ["tutorial", command]

            # Next the atom specifier
            target = query["target"][0]
            models = query["model"]
            if target == "sel":
                cmd_text.append("sel")
            elif target == "model":
                cmd_text.append(''.join(models))
            # else target must be "all":
            #   for which we leave off atom specifier completely

            # Then "highlight" specific parameters
            if command == "highlight":
                color = query["color"][0]
                cmd_text.append(color)
                count = query["count"][0]
                cmd_text.extend(["count", count])

            # Add remaining global options
            weighted = "weighted" in query
            cmd_text.extend(["weighted", "true" if weighted else "false"])
            transformed = "transformed" in query
            cmd_text.extend(["transformed", "true" if transformed else "false"])

            # Run the command
            cmd = ' '.join(cmd_text)
            from chimerax.core.commands import run
            run(self.session, cmd)
        else:
            from chimerax.core.errors import UserError
            raise UserError("unknown tutorial command: %s" % command)

    def update_models(self, trigger=None, trigger_data=None):
        # Update the <select> options in the web form with current
        # list of atomic structures.  Also enable/disable submit
        # buttons depending on whether there are any structures open.

        # Get the list of atomic structures
        from chimerax.core.atomic import AtomicStructure
        options = []
        for m in self.session.models:
            if not isinstance(m, AtomicStructure):
                continue
            spec = m.atomspec()
            options.append(("%s: %s" % (spec, m.name), spec))

        # Construct Javascript for updating <select> and submit buttons
        if not options:
            options_text = ""
            disabled_text = "true";
        else:
            options_text = ''.join(['<option value="%s">%s</option>' % (v, t)
                                    for t, v in options])
            disabled_text = "false";
        import json
        js = self.JSUpdate % (json.dumps(options_text), disabled_text)
        self.html_view.runJavaScript(js)

    JSUpdate = """
document.getElementById("model").innerHTML = %s;
var buttons = document.getElementsByClassName("submit");
for (var i = 0; i != buttons.length; ++i) {
    buttons[i].disabled = %s;
}
"""
