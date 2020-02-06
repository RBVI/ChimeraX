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
class OpenManager(ProviderManager):
    """Manager for open command"""

    def __init__(self, session):
        self.session = session
        self._openers = {}
        from chimerax.core.triggerset import TriggerSet
        self.triggers = TriggerSet()
        self.triggers.add_trigger("open command changed")
        """
        from chimerax.core import io
        for format_name, category, fmt_kw in io._used_register:
            self.add_format(format_name, category, **fmt_kw)
        io._user_register = self.add_format
        """

    def add_provider(self, bundle_info, name, *, type="open", want_path=False, check_path=True, **kw):
        logger = self.session.logger
        if kw:
            logger.warning("Open-command provider '%s' supplied unknown keywords in provider description: %s"
                % (name, repr(kw)))
        try:
            data_format = self.session.data_formats[name]
        except KeyError:
            raise ValueError("Open-command provider in bundle %s specified unknown data format '%s'"
                % (_readable_bundle_name(bundle_info), name))
        if data_format in self._openers:
            logger.warning("Replacing opener for '%s' from %s bundle with that from %s bundle"
                % (data_format.name, _readable_bundle_name(self._openers[data_format][0]),
                _readable_bundle_name(bundle_info)))
        self._openers[data_format] = (bundle_info, want_path, check_path)

    def end_providers(self):
        self.triggers.activate_trigger("open command changed", self)

def _readable_bundle_name(bundle_info):
    name = bundle_info.name
    if name.lower().startswith("chimerax"):
        return name[9:]
    return name
