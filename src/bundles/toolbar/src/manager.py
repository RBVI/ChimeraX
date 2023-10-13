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


class ToolbarManager(ProviderManager):
    """Manager for application toolbar"""

    def __init__(self, session, name):
        self.session = session
        self._toolbar = {}
        super().__init__(name)
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
        if not bundle_info.installed:
            return

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
        # <Provider tab="Markers" section="Place markers" name="pm1"
        #   mouse_mode="mark maximum" display_name="Maximum" description="Mark maximum"/>
        tab = kw.pop('tab', None)
        if tab is None:
            self.session.logger.warning('Missing tab %s' % where())
            return
        if tab == "Home":
            self.session.logger.warning('Home tab managed by user preferences %s' % where())
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
        if 'mouse_mode' in kw:
            if 'link' in kw:
                self.session.logger.warning('Mouse mode button can not be links %s' % where())
                return
            name = kw.pop("mouse_mode")
            display_name = kw.pop('display_name', name)
            description = kw.pop('description', None)
            icon = kw.pop('icon', None)
            if icon is not None:
                icon = bundle_info.get_path('icons/%s' % icon)
            bundle_info = fake_mouse_mode_bundle_info
        elif 'link' in kw:
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
            pi = bi.providers.get('toolbar/' + provider, None)
            if pi is None:
                self.session.logger.warning('Unable to find linked button %s' % where())
                return
            pi_kw = pi
            if 'mouse_mode' in pi_kw:
                self.session.logger.warning('Can not link to mouse mode buttons %s' % where())
                return
            display_name = kw.pop('display_name', None)
            if display_name is None:
                display_name = pi_kw.get("display_name", None)
                if display_name is None:
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
        else:
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
                # TODO: use default icon
        if name in section_dict:
            self.session.logger.warning('Overriding existing toolbar provider %s' % where())
        section_dict[display_name] = (name, bundle_info, icon, description, kw)
        if before is not None or after is not None:
            self._add_layout(section_dict, display_name, before, after)

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

    def set_enabled(self, enabled, tab_title, section_title, button_title):
        from . import tool
        tb = tool.get_toolbar_singleton(self.session, create=False)
        if tb:
            tb.set_enabled(enabled, tab_title, section_title, button_title)

    def show_group_button(self, tab_title, section_title, button_title):
        from . import tool
        tb = tool.get_toolbar_singleton(self.session, create=False)
        if tb:
            tb.show_group_button(tab_title, section_title, button_title)


class FakeMouseModeBundleInfo:
    # dummy to support mouse modes

    @staticmethod
    def run_provider(session, name, mgr, **kw):
        button_to_bind = 'right'
        from chimerax.core.commands import run
        if ' ' in name:
            name = '"%s"' % name
        run(session, f'ui mousemode {button_to_bind} {name}')


fake_mouse_mode_bundle_info = FakeMouseModeBundleInfo()
