# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2016 Regents of the University of California.
# All rights reserved.  This software provided pursuant to a
# license agreement containing restrictions on its disclosure,
# duplication and use.  For details see:
# https://www.rbvi.ucsf.edu/chimerax/docs/licensing.html
# This notice must be embedded in or attached to all copies,
# including partial copies, of the software or any revisions
# or derivations thereof.
# === UCSF ChimeraX Copyright ===

from chimerax.ui import HtmlToolInstance


class RegistrationUI(HtmlToolInstance):

    name = "Registration"

    SESSION_ENDURING = False
    SESSION_SAVE = False
    CUSTOM_SCHEME = "cxreg"

    PLACEMENT = None

    def __init__(self, session, ti):
        # Standard template stuff
        self.display_name = "ChimeraX Registration"
        super().__init__(session, ti.name, size_hint=(575, 400))

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
        dark_css = self.session.ui.dark_css()
        html = html.replace("DARK_CSS", dark_css)
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

    def handle_scheme(self, url):
        self.session.ui.thread_safe(self._register, url)

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
            "comment": ("Comment", ""),
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
        register(self.session, values["name"], values["email"],
                 organization=values["org"],
                 research=values["research"],
                 research_other=values["research_other"],
                 funding=values["funding"],
                 funding_other=values["funding_other"],
                 comment=values["comment"],
                 join_discussion=values["join_discussion"],
                 join_announcements=values["join_announcements"])
        self.delete()
