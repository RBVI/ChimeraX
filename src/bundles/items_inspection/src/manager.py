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
class ItemsInspection(ProviderManager):
    """Manager for options needed to inspect items"""

    def __init__(self, session):
        self.session = session
        self._item_info = {}
        from chimerax.core.triggerset import TriggerSet
        self.triggers = TriggerSet()
        self.triggers.add_trigger("inspection items changed")

    @property
    def item_types(self):
        return list(self._item_info.keys())

    def item_options(self, item_type):
        info = self._item_info[item_type]
        if not isinstance(info, (list, tuple)):
            info = self._item_info[item_type] = info.run_provider(self.session, item_type, self)
        return info[:]

    def add_provider(self, bundle_info, name, **kw):
        """ The provider's run_provider method should return a list of chimerax.ui.options classes
            that can be instantiated to inspect various properties of the item type.  The instances
            should have a "command_format" attribute that is a string with a single '%s', into which
            the "end user" inspector will interpolate command-line-target text (e.g., 'sel').  The
            result should be executable as a command when the option's value is changed to in turn
            accomplish the change in the data itself.
        """
        self._item_info[name] = bundle_info

    def end_providers(self):
        self.triggers.activate_trigger("inspection items changed", self)
