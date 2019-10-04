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

        def where():
            return 'for toolbar provider %r in bundle %r' % (name, bundle_info.name)
        # <Provider tab="Graphics" help="help:..."
        #   name="layout2" before="Nucleotides" after="Molecule Display"/>
        # <Provider tab="Molecule Display" section="Cartoons" help="help:..."
        #   name="layout1" before="Surfaces" after="Atoms"/>
        # <Provider tab="Home" section="Undo"
        #   name="Redo" icon="redo-variant.png" description="Redo last action"/>
        # <Provider tab="Home" section="Undo"
        #   name="Redo" link="BundleName:provider-name"/>
        tab = kw.pop('tab', None)
        if tab is None:
            self.session.logger.warning('Missing tab %s' % where())
            return
        section = kw.pop('section', None)
        before = kw.pop('before', None)
        after = kw.pop('after', None)
        help = kw.pop('help', None)
        if section is None:
            if before is None and after is None and help is None:
                self.session.logger.warning('Missing section %s' % where())
                return
            self._add_layout(self._toolbar, tab, before, after)
            # TODO: help
            return
        tab_dict = self._toolbar.setdefault(tab, {})
        section_dict = tab_dict.setdefault(section, {})
        if 'link' not in kw:
            display_name = kw.pop('display_name', None)
            icon = kw.pop('icon', None)
            description = kw.pop('description', None)
            if display_name is None and icon is None and description is None:
                self._add_layout(tab_dict, section, before, after)
                compact = kw.pop('compact', False)
                if compact:
                    section_dict['__compact__'] = True
                return
            if display_name is None:
                display_name = name
            if icon is not None:
                icon = bundle_info.get_path('icons/%s' % icon)
                if icon is None:
                    self.session.logger.warning('Unable to find icon %s' % where())
        else:
            link = kw.pop("link")
            try:
                bundle_name, provider = link.split(':', maxsplit=1)
            except ValueError as e:
                self.session.logger.warning('Unable to extract bundle name and its provider %s: %s' % (where(), e))
                return
            bi = self.session.toolshed.find_bundle(bundle_name, self.session.logger, installed=True)
            if bi is None:
                self.session.logger.warning('Uninstalled bundle %s' % where())
                return
            pi = bi.providers.get(provider, None)
            if pi is None:
                self.session.logger.warning('Unable to find linked button %s' % where())
                return
            pi_manager, pi_kw = pi
            if pi_manager != 'toolbar':  # double check that is a toolbar entry
                self.session.logger.warning('Linked button is not managed by "toolbar" %s' % where())
                return
            display_name = kw.pop('display_name', None)
            if display_name is None:
                display_name = pi_kw.get("display_name", None)
            if "display_name" is None:
                display_name = provider
            try:
                icon = pi_kw["icon"]
                description = pi_kw["description"]
            except KeyError as e:
                self.session.logger.warning('Missing %s in linked button %s' % (e, where()))
                return
            if icon is not None:
                icon = bi.get_path('icons/%s' % icon)
                if icon is None:
                    self.session.logger.warning('Unable to find icon %s' % where())
            name = provider
            bundle_info = bi
        if name in section_dict:
            self.session.logger.warning('Overriding existing toolbar provider %s' % where())
        section_dict[display_name] = (name, bundle_info, icon, description, kw)
        if before is not None or after is not None:
            self._add_layout(section_dict, name, before, after)

    def _add_layout(self, dict_, name, before, after):
        # layouts are an DAG
        layout = dict_.setdefault("__layout__", {})
        if after is not None:
            children = layout.setdefault(name, set())
            children.update(after.split(':'))
        if before is not None:
            for b in before.split(':'):
                children = layout.setdefault(b, set())
                children.add(name)

    def end_providers(self):
        # self.triggers.activate_trigger("toolbar changed", self)
        pass
