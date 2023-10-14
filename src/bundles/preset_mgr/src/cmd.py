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

from chimerax.core.errors import UserError

def preset_cmd(session, text1, text2=None):
    '''
    Apply preset depiction

    Args are either preset name or abbreviation (text2 == None),
    or category name / abbreviation + preset name / abbreviation (text2 != None)
    '''

    if text2 is None:
        cat_text = None
        preset_text = text1
    else:
        # RestOfLine arg doesn't strip quoting; do it by hand
        text2 = text2.strip()
        if text2[0] in ('"', "'") and text2[0] == text2[-1]:
            text2 = text2[1:-1]
        cat_text, preset_text = text1, text2

    preset_map = session.presets.presets_by_category
    if cat_text is None:
        match_all_presets(session, preset_text, preset_map)
    else:
        cat_matches = text_match(cat_text, preset_map.keys())
        if cat_matches:
            if len(cat_matches) > 1:
                raise UserError("Multiple preset category names match '%s': %s"
                    % (cat_text, '; '.join(cat_matches)))
            cat = cat_matches[0]
            matches = text_match(preset_text, preset_map[cat])
            if not matches:
                raise UserError("No preset name in category '%s' matches '%s'" % (cat, preset_text))
            if len(matches) > 1:
                raise UserError("Multiple preset names in category '%s' match '%s': %s"
                    % (cat, preset_text, '; '.join(matches)))
            run_preset(session, cat, matches[0])
        else:
            match_all_presets(session, cat_text + " " + preset_text, preset_map)

def match_all_presets(session, preset_text, preset_map):
    preset_names = [name for names in preset_map.values() for name in names]
    matches = text_match(preset_text, preset_names)
    if not matches:
        raise UserError("No preset name matches '%s'" % preset_text)
    if len(matches) > 1:
        raise UserError("Multiple preset names match '%s': %s" % (preset_text, '; '.join(matches)))
    for cat, presets in preset_map.items():
        if matches[0] in presets:
            run_preset(session, cat, matches[0])
            break
    else:
        raise AssertionError("Previously found preset '%s' no longer found" % matches[0])

def run_preset(session, category, preset):
    from chimerax.core.utils import titleize
    session.logger.info("Using preset: %s / %s" % (titleize(category), titleize(preset)))
    session.presets.preset_function(category, preset)()

def text_match(query, targets):
    # priority:  exact match;  begins with query text;  contains query text
    contains = []
    begins_with = []
    l_query = query.lower()
    for target in targets:
        l_target = target.lower()
        if l_query == l_target:
            return [target]
        if l_target.startswith(l_query):
            begins_with.append(target)
        elif l_query in l_target:
            contains.append(target)
    if begins_with:
        return begins_with
    return contains

def register_preset_command(logger):
    from chimerax.core.commands import CmdDesc, register, StringArg, RestOfLine
    desc = CmdDesc(
        required = [('text1', StringArg)],
        optional = [('text2', RestOfLine)],
        synopsis = 'apply preset depiction to models'
    )
    register('preset', desc, preset_cmd, logger=logger)
