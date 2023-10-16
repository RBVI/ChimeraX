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

def altlocs_change(session, alt_loc, residues=None):
    '''List altocs for residues

    Parameters
    ----------
    alt_loc : single character
        alt_loc to change to
    residues : sequence or Collection of Residues
        change the altlocs for theses residues.  If not specified, then all residues.
    '''
    if residues is None:
        from chimerax.atomic import all_residues
        residues = all_residues(session)

    from chimerax.core.errors import UserError
    if not residues:
        UserError("No residues specified")

    num_changed = num_found = 0
    for r in residues:
        r_locs = set(r.atoms.alt_locs)
        r_locs.discard(' ')
        if len(r_locs) == 1 and alt_loc in r_locs:
            # already set to that alt loc
            num_found += 1
            continue
        try:
            r.set_alt_loc(alt_loc)
        except ValueError:
            pass
        else:
            num_changed += 1
            num_found += 1
    unchanged = num_found - num_changed

    from chimerax.core.commands import plural_form, commas
    if num_found == 0:
        session.logger.warning("Alternate location %s not found in %d %s" % (alt_loc, len(residues),
            plural_form(residues, "residue")))
    elif len(residues) == 1 and num_changed == 1:
        session.logger.info("Changed %s to alternate location %s" % (list(residues)[0], alt_loc))
    elif unchanged == 0:
        session.logger.info("Changed %d %s to alternate location %s" % (num_changed,
            plural_form(num_changed, "residue"), alt_loc))
    else:
        session.logger.info("Changed %d %s to alternate location %s (%d %s were already %s)" % (num_changed,
            plural_form(num_changed, "residue"), alt_loc, unchanged, plural_form(unchanged, "residue"),
            alt_loc))

def altlocs_clean(session, residues=None):
    '''Change current alt locs into non-alt locs and remove non-current alt locs

    Parameters
    ----------
    residues : sequence or Collection of Residues
        'Clean' the altlocs for theses residues.  If not specified, then all residues.
    '''
    if residues is None:
        from chimerax.atomic import all_residues
        residues = all_residues(session)

    from chimerax.core.errors import UserError
    if not residues:
        UserError("No residues specified")

    num_cleaned = 0
    for r in residues:
        # r.atoms.altlocs is the current alt locs
        r_locs = set([al for a in r.atoms for al in a.alt_locs])
        r_locs.discard(' ')
        if r_locs:
            r.clean_alt_locs()
            num_cleaned += 1

    from chimerax.core.commands import plural_form, commas
    if num_cleaned == 0:
        session.logger.info("No alternate locations in %d %s" % (len(residues),
            plural_form(len(residues), "residue")))
    else:
        session.logger.info("Removed alternate locations from %d %s" % (num_cleaned,
            plural_form(num_cleaned, "residue")))

def altlocs_list(session, residues=None):
    '''List altocs for residues

    Parameters
    ----------
    residues : sequence or Collection of Residues
        List the altlocs for theses residues.  If not specified, then all residues.
    '''
    if residues is None:
        from chimerax.atomic import all_residues
        residues = all_residues(session)

    from chimerax.core.errors import UserError
    if not residues:
        UserError("No residues specified")

    residues = sorted(residues)
    no_alt_locs = 0
    alt_locs = []
    for r in residues:
        # r.atoms.altlocs is the current alt locs
        r_locs = set([al for a in r.atoms for al in a.alt_locs])
        r_locs.discard(' ')
        if r_locs:
            alt_locs.append((r, r_locs))
        else:
            no_alt_locs += 1

    from chimerax.core.commands import plural_form, commas
    if no_alt_locs:
        session.logger.info("%d %s %s no alternate locations" % (no_alt_locs,
            plural_form(no_alt_locs, "residue"), plural_form(no_alt_locs, "has", plural="have")))

    for r, r_locs in alt_locs:
        used = set(r.atoms.alt_locs)
        used.discard(' ')
        session.logger.info("%s has alternate locations %s (using %s)" % (r,
            commas(sorted(r_locs), conjunction="and"), commas(used, conjunction="and")))

def register_command(logger):
    from chimerax.core.commands import register, CmdDesc, AnnotationError, StringArg, Or, CharacterArg
    from chimerax.atomic import ResiduesArg

    desc = CmdDesc(required=[('alt_loc', CharacterArg)],
        optional = [('residues', ResiduesArg)],
        synopsis='change alternate atom locations')
    register('altlocs change', desc, altlocs_change, logger=logger)

    desc = CmdDesc(optional = [('residues', ResiduesArg)],
        synopsis='change current alternate atom locations into non-alt-locs and remove other alt locs')
    register('altlocs clean', desc, altlocs_clean, logger=logger)

    desc = CmdDesc(optional = [('residues', ResiduesArg)],
        synopsis='list alternate atom locations')
    register('altlocs list', desc, altlocs_list, logger=logger)
