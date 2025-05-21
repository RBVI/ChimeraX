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

fallbacks = { 'seq_viewer': 'seqview', 'large_align_viewer': 'grid', 'small_align_viewer': 'seqview' }
class _AlignmentsSettings(Settings):

    EXPLICIT_SAVE = {
        'seq_viewer': fallbacks['seq_viewer'],
        'large_align_viewer': fallbacks['large_align_viewer'],
        'large_align_threshold': 300,
        'small_align_viewer': fallbacks['small_align_viewer'],
        'assoc_error_rate': 10,
        'iterate': 2.0
    }

settings = None
def init(session):
    global settings
    # don't initialize a zillion times, which would also overwrite any changed but not
    # saved settings
    if settings is None:
        settings = _AlignmentsSettings(session, "alignments")

def register_settings_options(session):
    from chimerax.core.colors import color_name
    from chimerax.ui.options import IntOption, SymbolicEnumOption
    def viewer_option(variety, attribute):
        viewers = session.alignments.registered_viewers(variety)
        found_target = False
        for target in [getattr(settings, attribute), fallbacks[attribute]]:
            values = []
            for viewer in viewers:
                synonyms = session.alignments.viewer_info[variety][viewer]
                if target in synonyms:
                    values.append(target)
                    found_target = True
                else:
                    values.append(synonyms[0])
            if found_target:
                # class scope can't reference local variables, so...
                return lambda *args, v=values, l=[v.title() for v in viewers], **kw: \
                    SymbolicEnumOption(*args, values=v, labels=l, **kw)
    settings_info = {}
    settings_info['large_align_viewer'] = ("Large alignment viewer",
        viewer_option("alignment", "large_align_viewer"),
        "Default tool to use to show large (>= %d sequences) alignments" % settings.large_align_threshold)
    settings_info['large_align_threshold'] = (
        "Large alignment has \N{GREATER-THAN OR EQUAL TO} this many sequences",
        (IntOption, {'min': 2}), None)
    settings_info['small_align_viewer'] = ("Small alignment viewer",
        viewer_option("alignment", "small_align_viewer"),
        "Default tool to use to show small (< %d sequences) alignments" % settings.large_align_threshold)
    settings_info['seq_viewer'] = ("Single sequence viewer", viewer_option("sequence", "seq_viewer"),
        "Default tool to use to show a single sequence")
    for setting, setting_info in settings_info.items():
        opt_name, opt_class, balloon = setting_info
        if isinstance(opt_class, tuple):
            opt_class, kw = opt_class
        else:
            kw = {}
        opt = opt_class(opt_name, getattr(settings, setting), None,
            attr_name=setting, settings=settings, balloon=balloon, **kw)
        session.ui.main_window.add_settings_option("Sequences", opt)
