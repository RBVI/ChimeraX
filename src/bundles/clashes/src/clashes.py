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

from .settings import defaults

def find_clashes(session, test_atoms,
        assumed_max_vdw=2.1,
        attr_name=defaults["attr_name"],
        bond_separation=defaults["bond_separation"],
        clash_threshold=defaults["clash_threshold"],
        distance_only=None,
        hbond_allowance=defaults["clash_hbond_allowance"],
        ignore_hidden_models=False,
        inter_model=True,
        inter_submodel=False,
        intra_model=True,
        intra_res=False,
        intra_mol=True,
        res_separation=None,
        restrict="any"):
    """Detect steric clashes/contacts

       'test_atoms' should be an Atoms collection.

       'restrict' can be one of:
         - 'any':  interactions involving at least one atom from 'test_atoms' will be found
         - 'both':  interactions involving only atoms from 'test_atoms' will be found
         - 'cross':  interactions involving exactly one atom from 'test_atoms' will be found
         - an Atoms collection :  interactions between 'test_atoms' and the 'restrict' atoms will be found
       The "clash value" is the sum of the VDW radii minus the distance, which must exceed 'clash_threshold'.

       'hbond_allowance' is how much the clash value is reduced if one
       atom is a donor and the other an acceptor.

       If 'distance_only' is set (in which case it must be a positive numeric
       value), then both VDW radii, clash_threshold, and hbond_allowance are
       ignored and the center-center distance between the atoms must be <= the given value.

       Atom pairs are eliminated from consideration if they are less than
       or equal to 'bond_separation' bonds apart.

       Clashes involving hidden models ('visible' attr is False) are ignored
       if 'ignore_hidden_models' is True.
       Intra-residue clashes are ignored unless intra_res is True.
       Intra-model clashes are ignored unless intra_model is True.
       Intra-molecule (covalently connected fragment) clashes are ignored
       unless intra_mol is True.
       Inter-(sibling)submodel clashes are ignored unless inter_submodel is True.
       Inter-model clashes are ignored unless inter_model is True.

       If res_separation is not None, it should be a positive integer -- in which
       case for residues in the same chain, clashes/contacts are ignored unless
       the residues are at least that far apart in the sequence.

       Returns a dictionary keyed on atoms, with values that are
       dictionaries keyed on clashing atom with value being the clash value.
    """

    from chimerax.atomic import Structure
    use_scene_coords = inter_model and len(
        [m for m in session.models if isinstance(m, Structure)]) > 1
    # use the fast _closepoints module to cut down candidate atoms if we
    # can (since _closepoints doesn't know about "non-bonded" it isn't as
    # useful as it might otherwise be)
    if restrict == "any":
        if inter_model:
            from chimerax.atomic import all_atoms
            search_atoms = all_atoms(session)
        else:
            from chimerax.atomic import structure_atoms
            search_atoms = structure_atoms(test_atoms.unique_structures)
    elif restrict == "cross":
        if inter_model:
            from chimerax.atomic import all_atoms
            universe_atoms = all_atoms(session)
        else:
            from chimerax.atomic import structure_atoms
            universe_atoms = structure_atoms(test_atoms.unique_structures)
        other_atoms = universe_atoms.subtract(test_atoms)
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
        from chimerax.geometry import find_close_points
        t_close, o_close = find_close_points(test_coords, other_coords, cutoff)
        test_atoms = test_atoms[t_close]
        search_atoms = other_atoms[o_close]
    elif not isinstance(restrict, str):
        search_atoms = restrict
    else:
        search_atoms = test_atoms
    if ignore_hidden_models:
        test_atoms = test_atoms.filter(test_atoms.structures.visibles == True)
        search_atoms = search_atoms.filter(search_atoms.structures.visibles == True)

    if res_separation is not None:
        chain_pos = {}
        for s in test_atoms.unique_structures:
            for c in s.chains:
                for i, r in enumerate(c.residues):
                    if r:
                        chain_pos[r] = i
    from chimerax.atom_search import AtomSearchTree
    tree = AtomSearchTree(search_atoms, scene_coords=inter_model)
    clashes = {}
    from chimerax.geometry import distance
    intra_mol_map = {}
    for a in test_atoms:
        if distance_only:
            cutoff = distance_only
        else:
            cutoff = a.radius + assumed_max_vdw - clash_threshold
        crd = a.scene_coord if use_scene_coords else a.coord
        nearby = tree.search(crd, cutoff)
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
        if not intra_mol and a not in intra_mol_map:
            connected = set([a])
            to_do = list(a.neighbors)
            while to_do:
                conn = to_do.pop()
                connected.add(conn)
                for nb in conn.neighbors:
                    if nb not in connected:
                        to_do.append(nb)
            for ca in connected:
                intra_mol_map[ca] = connected
        for nb in nearby:
            if nb in exclusions:
                continue
            if not intra_res and a.residue == nb.residue:
                continue
            if not intra_mol and nb in intra_mol_map[a]:
                continue
            if not inter_model and a.structure != nb.structure:
                continue
            if not intra_model and a.structure == nb.structure:
                continue
            if a in clashes and nb in clashes[a]:
                continue
            if res_separation is not None:
                if a.residue.chain is not None and a.residue.chain == nb.residue.chain:
                    if abs(chain_pos[a.residue] - chain_pos[nb.residue]) < res_separation:
                        continue
            if not inter_submodel \
            and a.structure.id and nb.structure.id \
            and a.structure.id[0] == nb.structure.id[0] \
            and a.structure.id[:-1] == nb.structure.id[:-1] \
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

from chimerax.atomic import Element
hyd = Element.get_element(1)
negative = set([Element.get_element(sym) for sym in ["N", "O", "S"]])
from chimerax.atomic.idatm import type_info
def _donor(a):
    if a.element == hyd:
        if a.num_bonds > 0 and a.neighbors[0].element in negative:
            return True
    elif a.element in negative:
        try:
            if a.num_bonds < type_info[a.idatm_type].substituents:
                # implicit hydrogen
                return True
        except KeyError:
            pass
        for nb in a.neighbors:
            if nb.element == hyd:
                return True
    return False

def _acceptor(a):
    try:
        info = type_info[a.idatm_type]
    except KeyError:
        return False
    return info.substituents < info.geometry
