# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2022 Regents of the University of California. All rights reserved.
# The ChimeraX application is provided pursuant to the ChimeraX license
# agreement, which covers academic and commercial uses. For more details, see
# <https://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
#
# This particular file is part of the ChimeraX library. You can also
# redistribute and/or modify it under the terms of the GNU Lesser General
# Public License version 2.1 as published by the Free Software Foundation.
# For more details, see
# <https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html>
#
# THIS SOFTWARE IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND, EITHER
# EXPRESSED OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
# OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. ADDITIONAL LIABILITY
# LIMITATIONS ARE DESCRIBED IN THE GNU LESSER GENERAL PUBLIC LICENSE
# VERSION 2.1
#
# This notice must be embedded in or attached to all copies, including partial
# copies, of the software or any revisions or derivations thereof.
# === UCSF ChimeraX Copyright ===

from chimerax.core.settings import Settings

label_missing_attr = 'label_missing_structure_threshold'
class _AtomicSettings(Settings):
    EXPLICIT_SAVE = {
        'always_label_structure': False,
        'atomspec_contents': 'simple', # choices: simple, command (-line specifier), serial (number)
        label_missing_attr: 4,
    }

# 'settings' module attribute will be set by the initialization of the bundle API

def register_settings_options(session):
    from chimerax.ui.options import SymbolicEnumOption, IntOption, BooleanOption
    class AtomSpecOption(SymbolicEnumOption):
        values = ("command", "serial", "simple")
        labels = ("command line", "serial number", "simple")

    settings_info = {
        'always_label_structure': (
            "Balloon-help specifiers always show atomic model number",
            BooleanOption,
            "Always show structure model number in balloons and other output"
        ),
        'atomspec_contents': (
            "Balloon-help specifier style",
            AtomSpecOption,
            """How to format display of atomic data<br>
            <table>
            <tr><td>simple</td><td>&nbsp;</td><td>Simple readable form</td></tr>
            <tr><td>command line</td><td>&nbsp;</td><td>Form used in commands</td></tr>
            <tr><td>serial number</td><td>&nbsp;</td><td>Atom serial number</td></tr>
            </table>"""),
        label_missing_attr: (
            "Label missing-structure segments in models with \N{LESS-THAN OR EQUAL TO} N chains",
            (IntOption, {'min': 0}),
            "Label missing-structure pseudobonds with the number of residues that are missing"
            "\nif the structure has no more than N chains"),
    }
    for setting, setting_info in settings_info.items():
        opt_name, opt_class, balloon = setting_info
        if isinstance(opt_class, tuple):
            opt_class, kw = opt_class
        else:
            kw = {}
        opt = opt_class(opt_name, getattr(settings, setting), None,
            attr_name=setting, settings=settings, balloon=balloon, **kw)
        session.ui.main_window.add_settings_option("Labels", opt)
    from chimerax.core.commands import run
    def setting_changed(trig_name, trig_vals):
        setting_name, old, new = trig_vals
        if setting_name == label_missing_attr:
            run(session, "label missing %s" % new)
    settings.triggers.add_handler('setting changed', setting_changed)
