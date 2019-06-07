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

from chimerax.core.state import StateManager
class SchemesManager(StateManager):
    """Manager for html schemes used by all bundles"""

    def __init__(self, session):
        self.schemes = set()
        from chimerax.core.triggerset import TriggerSet
        self.triggers = TriggerSet()
        self.triggers.add_trigger("html schemes changed")

    def reset_state(self, session):
        pass

    def add_provider(self, bundle_info, name, **kw):
        self.schemes.add(name)

        def is_true(value):
            return value and value.casefold() in ('true', '1', 'on')

        from PyQt5.QtWebEngineCore import QWebEngineUrlScheme
        scheme = QWebEngineUrlScheme(name.encode('utf-8'))
        port = kw.get('defaultPort', None)
        if port is not None:
            scheme.setDefaultPort(int(port))
        syntax = kw.get('syntax', None)
        if syntax == "Path":
            scheme.setSyntax(QWebEngineUrlScheme.Syntax.Path)
        elif syntax == "Host":
            scheme.setSyntax(QWebEngineUrlScheme.Syntax.Host)
        elif syntax == "HostAndPort":
            scheme.setSyntax(QWebEngineUrlScheme.Syntax.HostAndPort)
        elif syntax == "HostPortAndUserInformation":
            scheme.setSyntax(QWebEngineUrlScheme.Syntax.HostPortAndUserInformation)
        flags = 0
        if is_true(kw.get("SecureScheme", None)):
            flags |= QWebEngineUrlScheme.SecureScheme
        if is_true(kw.get("LocalScheme", None)):
            flags |= QWebEngineUrlScheme.LocalScheme
        if is_true(kw.get("LocalAccessAllowed", None)):
            flags |= QWebEngineUrlScheme.LocalAccessAllowed
        if is_true(kw.get("NoAccessAllowed", None)):
            flags |= QWebEngineUrlScheme.NoAccessAllowed
        if is_true(kw.get("ServiceWorkersAllowed", None)):
            flags |= QWebEngineUrlScheme.ServiceWorkersAllowed
        if is_true(kw.get("ViewSourceAllowed", None)):
            flags |= QWebEngineUrlScheme.ViewSourceAllowed
        if is_true(kw.get("ContentSecurityPolicyIgnored", None)):
            flags |= QWebEngineUrlScheme.ContentSecurityPolicyIgnored
        if flags:
            scheme.setFlags(flags)
        QWebEngineUrlScheme.registerScheme(scheme)

    def end_providers(self):
        self.triggers.activate_trigger("html schemes changed", self)

    @staticmethod
    def restore_snapshot(session, data):
        return session.url_schemes

    def take_snapshot(self, session, flags):
        # Presets are "session enduring"
        return {}
