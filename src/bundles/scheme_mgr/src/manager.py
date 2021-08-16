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

from chimerax.core.toolshed import ProviderManager
class SchemesManager(ProviderManager):
    """Manager for html schemes used by all bundles"""

    def __init__(self, session, name):
        self.schemes = set()
        from chimerax.core.triggerset import TriggerSet
        self.triggers = TriggerSet()
        self.triggers.add_trigger("html schemes changed")
        super().__init__(name)

    def add_provider(self, bundle_info, name, **kw):
        if not bundle_info.installed:
            return
        self.schemes.add(name)

        def is_true(value):
            return value and value.casefold() in ('true', '1', 'on')

        from Qt import qt_have_web_engine
        if not qt_have_web_engine():
            return # macOS arm64 does not have QtWebEngine
    
        from Qt.QtWebEngineCore import QWebEngineUrlScheme
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
        if is_true(kw.get("CorsEnabled", None)):
            flags |= QWebEngineUrlScheme.CorsEnabled
        if flags:
            scheme.setFlags(flags)
        QWebEngineUrlScheme.registerScheme(scheme)

    def end_providers(self):
        self.triggers.activate_trigger("html schemes changed", self)
