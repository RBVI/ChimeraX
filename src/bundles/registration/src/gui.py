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

# ToolUI classes may also override
#   "delete" - called to clean up before instance is deleted
#
from chimerax.core.tools import ToolInstance

_EmptyPage = "<h2>Please select a chain and press <b>BLAST</b></h2>"
_InProgressPage = "<h2>BLAST search in progress&hellip;</h2>"


class RegistrationUI(ToolInstance):

    name = "Registration"

    SESSION_SAVE = False
    SESSION_ENDURING = False
    CUSTOM_SCHEME = "cxreg"

    def __init__(self, session, tool_name, blast_results=None, atomspec=None):
        # Standard template stuff
        ToolInstance.__init__(self, session, tool_name)
        self.display_name = "ChimeraX Registration"
        from chimerax.core.ui.gui import MainToolWindow
        self.tool_window = MainToolWindow(self)
        self.tool_window.manage(placement="side")
        parent = self.tool_window.ui_area

        # UI consists of a chain selector and search button on top
        # and HTML widget below for displaying results.
        # Layout all the widgets
        from PyQt5.QtWidgets import QGridLayout, QLabel, QComboBox, QPushButton
        from chimerax.core.ui.widgets import HtmlView
        layout = QGridLayout()
        self.html_view = HtmlView(parent, size_hint=(575, 700),
                                  interceptor=self._navigate,
                                  schemes=[self.CUSTOM_SCHEME])
        layout.addWidget(self.html_view, 0, 0)
        parent.setLayout(layout)

        # Fill in our registration form
        import os.path
        html_file = os.path.join(os.path.dirname(__file__),
                                 "registration_form.html")
        with open(html_file) as f:
            html = f.read()
        from .nag import check_registration
        from .cmd import ResearchAreas, FundingSources
        expiration = check_registration()
        if expiration is not None:
            exp_msg = ("<p>Your copy of ChimeraX is already registered "
                                "through %s.</p>" % expiration.strftime("%x"))
        else:
            exp_msg = "<p>Your copy of ChimeraX is unregistered.</p>"
        html = html.replace("EXPIRATION_PLACEHOLDER", exp_msg)
        html = html.replace("RESEARCH_PLACEHOLDER",
                            self._check_list("research", ResearchAreas, True))
        html = html.replace("FUNDING_PLACEHOLDER",
                            self._check_list("funding", FundingSources, False))
        self.html_view.setHtml(html)

    def _check_list(self, name, options, cap):
        lines = []
        for o in options:
            if o == "other":
                line = ('<input type="checkbox" '
                        'name="%s" value="%s">%s</input>&nbsp;'
                        '<input style="width:20em;" type="text" '
                        'name="%s_other"/>'
                        % (name, o, o.capitalize(), name))
            else:
                line = ('<input type="checkbox" name="%s" value="%s">%s</input>'
                        % (name, o, o if not cap else o.capitalize()))
            lines.append(line)
        return '<br/>'.join(lines)

    def _navigate(self, info):
        # "info" is an instance of QWebEngineUrlRequestInfo
        url = info.requestUrl()
        scheme = url.scheme()
        if scheme == self.CUSTOM_SCHEME:
            # self._load_pdb(url.path())
            self.session.ui.thread_safe(self._register, url)
        # For now, we only intercept our custom scheme.  All other
        # requests are processed normally.

    def _register(self, url):
        from urllib.parse import parse_qs
        query = parse_qs(url.query())
        fields = {
            "name": ("Name", None),
            "email": ("E-mail", None),
            "org": ("Organization", ""),
            "research": ("Research areas", []),
            "research_other": ("Other research area", ""),
            "funding": ("Funding sources", []),
            "funding_other": ("Other funding source", ""),
            "join_discussion": ("Join discussion mailing list", False),
            "join_announcements": ("Join announcements mailing list", False),
        }
        values = {}
        errors = []
        for name, info in fields.items():
            label, default = info
            try:
                value_list = query[name]
            except KeyError:
                if default is not None:
                    values[name] = default
                else:
                    errors.append("Field %r is required." % label)
            else:
                # We got a list of values (might be only one).
                # We use the default value type for some further processing.

                # If the default is boolean, make the value True.
                if default is False:
                    values[name] = True
                # If the default value is a list,  we use the supplied list.
                elif isinstance(default, list):
                    values[name] = value_list
                # If the default value is not a list, we use the last
                # supplied value.
                else:
                    values[name] = value_list[-1]
        if errors:
            self.session.logger.error('\n'.join(errors))
            return
        from .cmd import register
        register(self.session, values["name"], values["email"], values["org"],
                 values["research"], values["research_other"],
                 values["funding"], values["funding_other"],
                 join_discussion=values["join_discussion"],
                 join_announcements=values["join_announcements"])
