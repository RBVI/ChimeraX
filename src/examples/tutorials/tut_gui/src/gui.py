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
        super().__init__(session, ti.name, size_hint=(575, 200))
        self._build_ui()

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
        if command not in ["cofm", "highlight"]:
            from chimerax.errors import UserError
            raise UserError("unknown tutorial command: %s" % command)

        # Collect the optional parameters from URL query parameters
        from urllib.parse import parse_qs
        query = parse_qs(url.query())
        weighted = "weighted" in query
        transformed = "transformed" in query
        if command == "highlight":
            try:
                color = query["color"]
                count = query["count"]
                # TODO: map HTML color to ChimeraX color
                # TODO: convert "count" from string to integer
            except KeyError as e:
                from chimerax.errors import UserError
                raise UserError("parameter '%s' is missing" % e)

        # TODO: Construct a command and run it
        from chimerax.core.commands import run
