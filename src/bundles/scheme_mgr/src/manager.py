# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2022 Regents of the University of California. All rights reserved.
# This software is provided pursuant to the ChimeraX license agreement, which
# covers academic and commercial uses. For more information, see
# <http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
#
# This file is part of the ChimeraX library. You can also redistribute and/or
# modify it under the GNU Lesser General Public License version 2.1 as
# published by the Free Software Foundation. For more details, see
# <https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html>
#
# This file is distributed WITHOUT ANY WARRANTY; without even the implied
# warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. This notice
# must be embedded in or attached to all copies, including partial copies, of
# the software or any revisions or derivations thereof.
# === UCSF ChimeraX Copyright ===

from chimerax.core.toolshed import ProviderManager


class SchemesManager(ProviderManager):
    """Manager for html schemes used by all bundles"""

    def __init__(self, session, name):
        self.session = session
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
            return  # macOS arm64 does not have QtWebEngine

        from Qt.QtWebEngineCore import QWebEngineUrlScheme
        scheme = QWebEngineUrlScheme(name.encode('utf-8'))
        port = kw.get('defaultPort', None)
        if port is not None:
            del kw['defaultPort']
            scheme.setDefaultPort(int(port))
        syntax = kw.get('syntax', None)
        if syntax is not None:
            del kw['syntax']
            if syntax == "Path":
                scheme.setSyntax(QWebEngineUrlScheme.Syntax.Path)
            elif syntax == "Host":
                scheme.setSyntax(QWebEngineUrlScheme.Syntax.Host)
            elif syntax == "HostAndPort":
                scheme.setSyntax(QWebEngineUrlScheme.Syntax.HostAndPort)
            elif syntax == "HostPortAndUserInformation":
                scheme.setSyntax(QWebEngineUrlScheme.Syntax.HostPortAndUserInformation)
        flags = QWebEngineUrlScheme.Flag(0)
        if is_true(kw.get("SecureScheme", None)):
            del kw["SecureScheme"]
            flags |= QWebEngineUrlScheme.Flag.SecureScheme
        if is_true(kw.get("LocalScheme", None)):
            del kw["LocalScheme"]
            flags |= QWebEngineUrlScheme.Flag.LocalScheme
        if is_true(kw.get("LocalAccessAllowed", None)):
            del kw["LocalAccessAllowed"]
            flags |= QWebEngineUrlScheme.Flag.LocalAccessAllowed
        if is_true(kw.get("NoAccessAllowed", None)):
            del kw["NoAccessAllowed"]
            flags |= QWebEngineUrlScheme.Flag.NoAccessAllowed
        if is_true(kw.get("ServiceWorkersAllowed", None)):
            del kw["ServiceWorkersAllowed"]
            flags |= QWebEngineUrlScheme.Flag.ServiceWorkersAllowed
        if is_true(kw.get("ViewSourceAllowed", None)):
            del kw["ViewSourceAllowed"]
            flags |= QWebEngineUrlScheme.Flag.ViewSourceAllowed
        if is_true(kw.get("ContentSecurityPolicyIgnored", None)):
            del kw["ContentSecurityPolicyIgnored"]
            flags |= QWebEngineUrlScheme.Flag.ContentSecurityPolicyIgnored
        if is_true(kw.get("CorsEnabled", None)):
            del kw["CorsEnabled"]
            flags |= QWebEngineUrlScheme.Flag.CorsEnabled
        if flags:
            scheme.setFlags(flags)
        if kw:
            self.session.logger.warning(
                f"unknown keywords for {self.name}'s {name}: {' '.join(kw)}")
        QWebEngineUrlScheme.registerScheme(scheme)

    def end_providers(self):
        self.triggers.activate_trigger("html schemes changed", self)
