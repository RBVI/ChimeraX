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

# providers return this
import abc
class RenderAttrInfo(metaclass=abc.ABCMeta):
    """Info needed for rendering attributes for instances of a class"""
    def __init__(self, session):
        self.session = session

    @abc.abstractproperty
    def class_object(self):
        """Return the class object (which must offer the core attribute-registration API)"""
        pass

    def hide_attr(self, attr_name, rendering):
        """Return True if attr_name should not be shown by the Render/Select tab of the tool
          (respectively rendering True/False).
        """
        return (attr_name.startswith("num_") or attr_name == "number") and rendering

    @abc.abstractmethod
    def model_filter(self, model):
        """When this class is selected, should the given model be shown in the model list?"""
        pass

    @abc.abstractmethod
    def render(self, session, attr_name, models, method, parameters, selected_only):
        """Render the given models based on attr_name as requested.

        The only current method is 'color', for which 'parameters' is a two-tuple of a set of
        targets to color (of those legal from the Provider declaration) and a sequence of value-RGBA
        pairs (RGBA channels in 0-1 range) for the color to use at that value.  One of the values may
        be None, in which case None values should receive that color.  If 'selected_only' is True, then
        the coloring should only be applied to selected instances.

        This method should carry out the rendering using a command if possible.
        """
        pass

    @abc.abstractmethod
    def values(self, attr_name, models):
        """Get the values of the given attribute in the given models.  Returns a two-tuple,
        the first component of which is a sequence of the non-None values, and the second is
        a boolean indicating if there were any None values.
        """
        pass

from chimerax.core.toolshed import ProviderManager

class RenderByAttrManager(ProviderManager):

    def __init__(self, session):
        self._initializing = True
        self.session = session
        self.attr_classes = {}
        self._provider_bundles = {}
        self._ui_names = {}
        self._color_targets = {}
        super().__init__("render by attribute")
        self._initializing = False

    def add_provider(self, bundle_info, name, *,
            ui_name=None, colors_atoms=True, colors_cartoons=True, colors_surfaces=True):
        # 'name' is the name used as an arg in the command
        # 'ui_name' is the name used in the tool interface
        self._provider_bundles[name] = bundle_info
        self._ui_names[name] = name if ui_name is None else ui_name
        self._color_targets[name] = color_targets = set()
        if colors_atoms != "false":
            color_targets.add("atoms")
        if colors_cartoons != "false":
            color_targets.add("cartoons")
        if colors_surfaces != "false":
            color_targets.add("surfaces")
        self._infos = {}

    def color_targets(self, provider_name):
        return self._color_targets[provider_name]

    def end_providers(self):
        if self._initializing:
            return
        from .tool import RenderByAttrTool
        for tool in self.session.tools.find_by_class(RenderByAttrTool):
            tool._new_classes()

    @property
    def provider_names(self):
        return list(self._provider_bundles.keys())

    def render_attr_info(self, provider_name):
        if provider_name not in self._infos:
            self._infos[provider_name] = self._provider_bundles[provider_name].run_provider(
                self.session, provider_name, self)
        return self._infos[provider_name]

    def ui_name(self, provider_name):
        return self._ui_names[provider_name]

_manager = None
def get_manager(session):
    global _manager
    if _manager is None:
        _manager = RenderByAttrManager(session)
    return _manager
