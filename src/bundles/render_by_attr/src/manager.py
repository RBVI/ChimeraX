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

# providers return this
import abc
class RenderAttrInfo(metaclass=abc.ABCMeta):
    def __init__(self, session):
        self.session = session

    @abc.abstractproperty
    def class_objects(self):
        # Can be more than one class since the Render By Attr tool may want to combine classes
        # together (e.g. Structure and AtomicStructure) for user interface purposes
        pass

from chimerax.core.toolshed import ProviderManager

class RenderByAttrManager(ProviderManager):

    def __init__(self, session):
        self.session = session
        self.attr_classes = {}
        self._provider_bundles = {}
        self._ui_names = {}
        super().__init__("render by attribute")

    def add_provider(self, bundle_info, name, *, ui_name=None, indirect=False, new_model_only=False,
            auto_style=True):
        # 'name' is the name used as an arg in the command
        # 'ui_name' is the name used in the tool interface
        self._provider_bundles[name] = bundle_info
        self._ui_names[name] = name if ui_name is None else ui_name

    def attr_class(self, provider_name):
        if provider_name not in self.attr_classes:
            self.attr_classes[provider_name] = self._provider_bundles[provider_name].run_provider(
                self.session, provider_name, self)
        return self.attr_classes[provider_name]

    def end_providers(self):
        from .tool import RenderByAttrTool
        for tool in self.session.tools.find_by_class(RenderByAttrTool):
            tool._new_classes()

    @property
    def provider_names(self):
        return list(self._provider_bundles.keys())

    def ui_name(self, provider_name):
        return self._ui_names[provider_name]

_manager = None
def get_manager(session):
    global _manager
    if _manager is None:
        _manager = RenderByAttrManager(session)
    return _manager
