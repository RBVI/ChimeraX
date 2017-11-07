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

from .settings import defaults

def find_clashes(session, test_atoms,
        assumed_max_vdw=2.1,
        attr_name=defaults["attr_name"],
        bond_separation=defaults["bond_separation"],
        clash_threshold=defaults["clash_threshold"],
        distance_only=None,
        group_name=defaults["group_name"],
        hbond_allowance=defaults["clash_hbond_allowance"],
        inter_model=True,
        inter_submodel=False,
        intra_res=False,
        intra_mol=True,
        test="others"):
    """Detect steric clashes/contacts

       'test_atoms' should be an Atoms collection.

       If 'test' is 'others' then non-bonded clashes between atoms in
       'test_atoms' and non-'test_atoms' atoms will be found.  However, if
       there are no non-'test_atoms' (i.e. test_atoms is all the atoms that
       exist), then 'others' will be treated as 'self'.  If 'test' is 'self'
       then non-bonded clashes within 'test_atoms' atoms will be found.
       Otherwise 'test' should be a list of atoms to test against.
       The "clash value" is the sum of the VDW radii minus the distance,
       keeping only the maximal clash (which must exceed 'clash_threshold').

       'hbond_allowance' is how much the clash value is reduced if one
       atom is a donor and the other an acceptor.

       If 'distance_only' is set (in which case it must be a positive numeric
       value), then both VDW radii, clash_threshold, and hbond_allowance are
       ignored and the center-center distance between the atoms must <= the given value.

       Atom pairs are eliminated from consideration if they are less than
       or equal to 'bond_separation' bonds apart.

       Intra-residue clashes are ignored unless intra_res is True.
       Intra-molecule (covalently connected fragment) clashes are ignored
       unless intra_mol is True.
       Inter-submodel clashes are ignored unless inter_submodel is True.
       Inter-model clashes are ignored unless inter_model is True.

       Returns a dictionary keyed on atoms, with values that are
       dictionaries keyed on clashing atom with value being the clash value.
    """

    from chimerax.core.atomic import Structure
    use_scene_coords = inter_model and len(
        [m for m in session.models if isinstance(m, Structure)]) > 1
    # use the fast _closepoints module to cut down candidate atoms if we
    # can (since _closepoints doesn't know about "non-bonded" it isn't as
    # useful as it might otherwise be)
    if test == "others":
        if inter_model:
            from chimerax.core.atomic import all_atoms
            universe_atoms = all_atoms(session)
        else:
            from chimerax.core.atomic import structure_atoms
            universe_atoms = structure_atoms(test_atoms.unique_structures)
        other_atoms = universe_atoms.subtract(test_atoms)
        if len(other_atoms) == 0:
            # no other atoms, change test to "self"
            test = "self"
            search_atoms = test_atoms
        else:
            if distance_only:
                cutoff = distance_only
            else:
                cutoff = 2.0 * assumed_max_vdw - clash_threshold
            if use_scene_coords:
                test_coords = test_atoms.scene_coords
                other_coords = other_atoms.scene_coords
            else:
                test_coords = test_atoms.coords
                other_coords = other_atoms.coords
            from chimerax.core.geometry import find_close_points
            t_close, o_close = find_close_points(test_coords, other_coords, cutoff)
            test_atoms = test_atoms[t_close]
            search_atoms = other_atoms[o_close]
    elif not isinstance(test, str):
        search_atoms = test
    else:
        search_atoms = test_atoms

    from chimerax.core.atomic import atom_search_tree
    tree = atom_search_tree(search_atoms, scene_coords=inter_model)
    clashes = {}
    from chimerax.core.geometry import distance
    for a in test_atoms:
        if distance_only:
            cutoff = distance_only
        else:
            cutoff = a.radius + assumed_max_vdw - clash_threshold
        crd = a.scene_coord if use_scene_coords else a.coord
        nearby = tree.search_tree(crd, cutoff)
        if not nearby:
            continue
        need_expansion = [a]
        exclusions = set(need_expansion)
        for i in range(bond_separation):
            next_need = []
            for expand in need_expansion:
                for n in expand.neighbors:
                    if n in exclusions:
                        continue
                    exclusions.add(n)
                    next_need.append(n)
            need_expansion = next_need
        for nb in nearby:
            if nb in exclusions:
                continue
            if not intra_res and a.residue == nb.residue:
                continue
            if not intra_mol and a.molecule.rootForAtom(a,
                    True) == nb.molecule.rootForAtom(nb, True):
                continue
            if not inter_model and a.structure != nb.structure:
                continue
            if a in clashes and nb in clashes[a]:
                continue
            if not inter_submodel \
            and a.structure.id[0] == nb.structure.id[0] \
            and a.structure.id[1:] != nb.structure.id[1:]:
                continue
            if use_scene_coords:
                a_crd, nb_crd = a.scene_coord, nb.scene_coord
            else:
                a_crd, nb_crd = a.coord, nb.coord
            if distance_only:
                clash = distance_only - distance(a_crd, nb_crd)
            else:
                clash = a.radius + nb.radius - distance(a_crd, nb_crd)
            if hbond_allowance and not distance_only:
                if (_donor(a) and _acceptor(nb)) or (_donor(nb) and _acceptor(a)):
                    clash -= hbond_allowance
            if distance_only:
                if clash < 0.0:
                    continue
            elif clash < clash_threshold:
                continue
            clashes.setdefault(a, {})[nb] = clash
            clashes.setdefault(nb, {})[a] = clash
    return clashes

negative = set(["N", "O", "S"])
from chimerax.core.atomic.idatm import type_info
def _donor(a):
    if a.element.number == 1:
        if a.neighbors and a.neighbors[0].element.name in negative:
            return True
    elif a.element.name in negative:
        try:
            if len(a.bonds) < type_info[a.idatm_type].substituents:
                # implicit hydrogen
                return True
        except KeyError:
            pass
        for nb in a.neighbors:
            if nb.element.number == 1:
                return True
    return False

def _acceptor(a):
    try:
        info = type_info[a.idatm_type]
    except KeyError:
        return False
    return info.substituents < info.geometry
