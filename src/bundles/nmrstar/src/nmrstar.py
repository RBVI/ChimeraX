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

"""
NMRSTAR: Read NMR-STAR files
============================

Read distance restraints from NMR-STAR files and show them as pseudobonds on an atomic model.
"""

# -----------------------------------------------------------------------------
#
def read_nmr_star(session, path, name,
                  structures = None, type = None,
                  color = (160,255,160,255), radius = 0.05,
                  long_color = (255,255,0,255), long_radius = 0.2, dashes = 0):

    import pynmrstar
    entry = pynmrstar.Entry.from_file(path)

    constraint_tag_names = ['Auth_asym_ID_1', 'Auth_seq_ID_1', 'Auth_comp_ID_1', 'Auth_atom_ID_1',
                            'Auth_asym_ID_2', 'Auth_seq_ID_2', 'Auth_comp_ID_2', 'Auth_atom_ID_2',
                            'Distance_lower_bound_val', 'Distance_upper_bound_val']
    constraint_sets = []
    for saveframe in entry.get_saveframes_by_category('general_distance_constraints'):
        constraint_type = ','.join(saveframe.get_tag('Constraint_type'))
        if type is not None and constraint_type != type:
            continue
        constraint_loop = saveframe.get_loop('Gen_dist_constraint')
        clist = constraint_loop.get_tag(constraint_tag_names)
        clist = [(cid1, int(rnum1), rname1, atom1, cid2, int(rnum2), rname2, atom2,
                  (None if dist_min == '.' else float(dist_min)),
                  (None if dist_max == '.' else float(dist_max)))
                 for cid1, rnum1, rname1, atom1, cid2, rnum2, rname2, atom2, dist_min, dist_max in clist]
        constraint_sets.append((constraint_type, clist))

    if structures is None:
        structures = _structures_with_constraint_atoms(session, constraint_sets)

    if structures:
        _save_distance_limits_in_session(session)
        for structure in structures:
            for ctype, clist in constraint_sets:
                pbgroup = _make_constraint_pseudobonds(structure, ctype, clist,
                                                       color, radius, long_color, long_radius)
                if dashes is not None and pbgroup is not None:
                    pbgroup.dashes = dashes

            
    descrip = ', '.join([f'{len(constraint_sets)} constraint lists']
                        + [f'{len(clist)} {ctype}' for ctype, clist in constraint_sets])
    from os.path import basename
    msg = f'Read NMR-STAR {basename(path)}, {descrip} applied to {len(structures)} structures'
    
    return [], msg	# Don't return models since they are already added to session

# -----------------------------------------------------------------------------
#
def _make_constraint_pseudobonds(structure, ctype, clist, color, radius, long_color, long_radius):

    aplist = []
    missing = set()
    mcount = 0
    atom_table = {(a.residue.chain_id, a.residue.number, a.name):a for a in structure.atoms}
    for cid1, rnum1, rname1, atom1, cid2, rnum2, rname2, atom2, dist_min, dist_max in clist:
        a1 = atom_table.get((cid1, rnum1, atom1))
        if a1 is None:
            missing.add((cid1, rnum1, atom1))
            mcount += 1
            continue
        a2 = atom_table.get((cid2, rnum2, atom2))
        if a2 is None:
            missing.add((cid2, rnum2, atom2))
            mcount += 1
            continue
        aplist.append((a1, a2, dist_min, dist_max))

    if missing:
        matoms = ','.join(f'/{cid}:{rnum}@{aname}' for cid, rnum, aname in sorted(missing))
        log = structure.session.logger
        log.warning(f'Missing {len(missing)} atoms in {mcount} of {len(clist)} {ctype} constraints: {matoms}')

    if len(aplist) == 0:
        return None
    
    gname = ctype + ' constraints'
    g = structure.pseudobond_group(gname, create_type = None)
    if g is not None:
        # Pseudobond group with this name already exists.  Add a suffix
        i = 2
        while structure.pseudobond_group(gname + f' {i}', create_type = None):
            i += 1
        gname = gname + f' {i}'
    g = structure.pseudobond_group(gname)

    for a1, a2, dist_min, dist_max in aplist:
        b = g.new_pseudobond(a1, a2)
        if dist_max is not None and b.length > dist_max:
            b.color = long_color
            b.radius = long_radius
        else:
            b.color = color
            b.radius = radius
        b.nmr_min_distance = dist_min
        b.nmr_max_distance = dist_max

    return g

# -----------------------------------------------------------------------------
#
def _structures_with_constraint_atoms(session, constraint_sets):
    atom_ids = set()
    for ctype, clist in constraint_sets:
        for cid1, rnum1, rname1, atom1, cid2, rnum2, rname2, atom2, dist_min, dist_max in clist:
            atom_ids.add((cid1, rnum1, rname1, atom1))
            atom_ids.add((cid2, rnum2, rname2, atom2))

    from chimerax.atomic import all_atomic_structures
    structures = [m for m in all_atomic_structures(session) if _have_atoms(m.atoms, atom_ids)]
    return structures

# -----------------------------------------------------------------------------
#
def _have_atoms(atoms, atom_ids):
    have_ids = set()
    for a in atoms:
        r = a.residue
        have_ids.add((r.chain_id, r.number, r.name, a.name))
    for aid in atom_ids:
        if aid not in have_ids:
            return False
    return True

# -----------------------------------------------------------------------------
#
def _save_distance_limits_in_session(session):
    'Add pseudobond nmr_max_distance and nmr_min_distance attributes for session saving.'
    from chimerax.atomic import Pseudobond
    Pseudobond.register_attr(session, "nmr_min_distance", "NMR-STAR", attr_type=float)
    Pseudobond.register_attr(session, "nmr_max_distance", "NMR-STAR", attr_type=float)

# -----------------------------------------------------------------------------
#
def pdb_fetch(session, id, ignore_cache=False,
              structures = None, type = None,
              color = (160,255,160,255), radius = 0.05,
              long_color = (255,255,0,255), long_radius = 0.2, dashes = 0):
  '''
  Fetch NMR-STAR constraint file from PDB.

      https://files.wwpdb.org/pub/pdb/data/structures/divided/nmr_data/bf/8bfg_nmr-data.str.gz
  '''

  url_pattern = 'https://files.wwpdb.org/pub/pdb/data/structures/divided/nmr_data/%s/%s.gz'

  id = id.lower()
  
  # Fetch NMR-STAR data from PDB.
  log = session.logger
  log.status('Fetching %s from PDB...' % (id,))

  file_name = f'{id}_nmr-data.str'
  file_url = url_pattern % (id[1:3], file_name)

  from chimerax.core.fetch import fetch_file
  path = fetch_file(session, file_url, f'NMR restraints {id}', file_name, 'PDB',
                    ignore_cache=ignore_cache)

  models, status = read_nmr_star(session, path, file_name,
                                 structures = structures, type = type,
                                 color = color, radius = radius,
                                 long_color = long_color, long_radius = long_radius,
                                 dashes = dashes)
  return models, status
