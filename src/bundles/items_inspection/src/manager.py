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
class ItemsInspection(ProviderManager):
    """Manager for options needed to inspect items"""

    def __init__(self, session):
        self.session = session
        self._item_info = {}
        self._ui_names = {}
        from chimerax.core.triggerset import TriggerSet
        self.triggers = TriggerSet()
        self.triggers.add_trigger("inspection items changed")
        super().__init__("items inspection")

    @property
    def item_types(self):
        return list(self._item_info.keys())

    def item_info(self, item_type):
        info = self._item_info[item_type]
        if not isinstance(info, (list, tuple)):
            info = self._item_info[item_type] = info.run_provider(self.session, item_type, self)
        return info[:]

    def add_provider(self, bundle_info, name, *, ui_name=None, **kw):
        """ The provider's run_provider method should return a list of 2-tuples.  The first member of each
            tuple should be a chimerax.ui.options class that can be instantiated to inspect a property of
            the item type.  The resulting instance should have a "command_format" attribute that is a
            string with a single '%s', into which the "end user" inspector will interpolate command-line-
            target text (e.g., 'sel').  The result should be executable as a command when the option's
            value is changed to in turn accomplish the change in the data itself.  The second member of
            the tuple provides information on when the option need updating from external changes.
            It is a (trigger set, trigger name, boolean func) tuple for a trigger that fires when
            relevant changes occur.  The boolean function will be called with the trigger's data
            and should return True when items of the relevant type have been modified.

            If an option controls a particular attribute of the item then the option's 'attr_name'
            attribute should be set to that.  This will cause the option's balloon help to automatically
            add information about the attribute name to the bottom of the help balloon.  If the
            values of the attributes need explanation (e.g. they're an integer enumeration with
            semantic meaning), then set the option's 'attr_values_balloon' attribute to whatever
            additional text you would want to add to the bottom of the balloon.

            Since any inspector would have no idea what the correct default is to provide to the
            option constructor, the option needs to have its 'default' class attribute set in the
            class definition (unless a default of None is acceptable, which is what is provided to
            the constructor).

            If an item wants to present a different name in user interfaces than 'name', then ui_name
            should be specified in its Provider tag.
        """
        self._item_info[name] = bundle_info
        self._ui_names[name] = name if ui_name is None else ui_name

    def end_providers(self):
        self.triggers.activate_trigger("inspection items changed", self)

    def ui_name(self, item_type):
        return self._ui_names[item_type]
