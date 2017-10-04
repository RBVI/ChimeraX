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
        from .cmd import OrganizationTypes, UsageTypes
        expiration = check_registration()
        if expiration is not None:
            exp_msg = ("<p>Your copy of ChimeraX is already registered "
                                "through %s.</p>" % expiration.strftime("%x"))
        else:
            exp_msg = "<p>Your copy of ChimeraX is unregistered.</p>"
        org_list = ['<input type="radio" name="org_type" value="%s">%s</input>'
                    % (ot, ot.capitalize()) for ot in OrganizationTypes]
        usage_list = ['<input type="radio" name="usage" value="%s">%s'
                      % (u, u.capitalize()) for u in UsageTypes]
        html = html.replace("EXPIRATION_PLACEHOLDER", exp_msg)
        html = html.replace("ORGTYPE_PLACEHOLDER", ' '.join(org_list))
        html = html.replace("USAGE_PLACEHOLDER", ' '.join(usage_list))
        self.html_view.setHtml(html)

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
            "org_type": ("Organization type", ""),
            "usage": ("Primary usage", ""),
            "nih_funded": ("Funded by NIH", False),
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
                # Special processing for boolean values.
                # If checkbox is unchecked, we do not get
                # anything in the query so we get the default
                # False value.  If we get anything, we turn
                # it into True.
                if default is False:
                    values[name] = True
                else:
                    values[name] = value_list[-1]
        if errors:
            self.session.logger.error('\n'.join(errors))
            return
        from .cmd import register
        register(self.session, values["name"], values["email"], values["org"],
                 values["org_type"], values["usage"], values["nih_funded"],
                 join_discussion=values["join_discussion"],
                 join_announcements=values["join_announcements"])
