# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2019 Regents of the University of California.
# All rights reserved.  This software provided pursuant to a
# license agreement containing restrictions on its disclosure,
# duplication and use.  For details see:
# http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html
# This notice must be embedded in or attached to all copies,
# including partial copies, of the software or any revisions
# or derivations thereof.
# === UCSF ChimeraX Copyright ===

from chimerax.core.toolshed import ProviderManager


class ToolbarManager(ProviderManager):
    """Manager for application toolbar"""

    def __init__(self, session):
        self.session = session
        self._toolbar = {}
        return
        # TODO:
        from . import settings
        settings.settings = settings._ToolbarSettings(session, "toolbar")
        from chimerax.core.triggerset import TriggerSet
        self.triggers = TriggerSet()
        self.triggers.add_trigger("toolbar changed")
        if session.ui.is_gui:
            session.ui.triggers.add_handler(
                'ready',
                lambda *arg, ses=session: settings.register_settings_options(session))

    def clear(self):
        self._toolbar.clear()

    def add_provider(self, bundle_info, name, **kw):
        if name == "__tab__":
            # <Provider tab="Home" section="File"
            #   name="__tab__" layout="Open:Save"/>
            # <Provider tab="Home" before="Molecule Display"
            #   name="__tab__" layout="File:Images:Atoms:Cartoons:Styles:Background:Lighting"/>
            tab = kw.pop('tab', None)
            if tab is None:
                self.session.logger.warning(
                    'Missing toolbar tab name for toolbar provider %r in bundle %r' % (
                        name, bundle_info.name))
                return
            url = kw.pop('help', None)
            before = kw.pop('before', None)
            layout = kw.pop('layout', None)
            # TODO
            return
        # <Provider tab="Home" section="Undo"
        #   name="Redo" icon="redo-variant.png" description="Redo last action"/>
        # <Provider tab="Home" section="Undo"
        #   name="Redo" link="BundleName:provider-name"/>
        tab = kw.pop('tab', None)
        section = kw.pop('section', None)
        if tab is None or section is None:
            self.session.logger.warning('Failed to add toolbar provider %r' % name)
            return
        tab_dict = self._toolbar.setdefault(tab, {})
        section_dict = tab_dict.setdefault(section, {})
        if 'link' not in kw:
            display_name = kw.pop('display_name', None)
            if display_name is None:
                display_name = name
            icon = kw.pop('icon', None)
            description = kw.pop('description', None)
            if icon is not None:
                icon = bundle_info.get_path('icons/%s' % icon)
                if icon is None:
                    self.session.logger.warning('Unable to find icon for toolbar provider %r' % name)
        else:
            link = kw.pop("link")
            try:
                bundle_name, provider = link.split(':', maxsplit=1)
            except ValueError as e:
                self.session.logger.warning('Unable to extract bundle name and its provider from toolbar provider %r: %s' % (name, e))
                return
            bi = self.session.toolshed.find_bundle(bundle_name, self.session.logger, installed=True)
            if bi is None:
                self.session.logger.warning('Uninstalled bundle in toolbar provider %r' % name)
                return
            pi = bi.providers.get(provider, None)
            if pi is None:
                self.session.logger.warning('Unable to find linked button in toolbar provider %r' % name)
                return
            pi_manager, pi_kw = pi
            if pi_manager != 'toolbar':  # double check that is a toolbar entry
                self.session.logger.warning('Linked button is not managed by "toolbar" in toolbar provider %r' % name)
                return
            try:
                display_name = pi_kw["display_name"]
                icon = pi_kw["icon"]
                description = pi_kw["description"]
            except KeyError as e:
                self.session.logger.warning('Missing %s for linked button in toolbar provider %r' % (e, name))
                return
            if icon is not None:
                icon = bundle_info.get_path('icons/%s' % icon)
                if icon is None:
                    self.session.logger.warning('Unable to find icon for toolbar provider %r' % name)
            name = provider
            bundle_info = bi
        if name in section_dict:
            self.session.logger.warning('Overriding existing toolbar provider %r' % name)
        section_dict[display_name] = (name, bundle_info, icon, description, kw)

    def end_providers(self):
        # self.triggers.activate_trigger("toolbar changed", self)
        pass
