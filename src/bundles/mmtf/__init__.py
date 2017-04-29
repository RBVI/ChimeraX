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

from chimerax.core.toolshed import BundleAPI
from chimerax.core.atomic import AtomicStructure
import mmtf

HELIX = 'helix'
STRAND = 'strand'


class _MyAPI(BundleAPI):

    @staticmethod
    def fetch_from_database(session, identifier, ignore_cache=False, database_name=None):
        # 'fetch_from_database' is called by session code to fetch data with give identifier
        # returns (list of models, status message)
        return fetch_mmtf(session, identifier)

bundle_api = _MyAPI()

# See README.rst for documentation and implementation status


def fetch_mmtf(session, pdb_id):
    # TODO: cache?
    data = mmtf.fetch(pdb_id)

    # Data structure traversal based on example given at
    # https://github.com/rcsb/mmtf/blob/v0.2/spec.md#traversal

    # detect optional data
    has_bond_atom_list = hasattr(data, 'bond_atom_list')
    has_bond_order_list = hasattr(data, 'bond_order_list')
    has_b_factor_list = hasattr(data, 'b_factor_list')
    has_occupancy_list = hasattr(data, 'occupancy_list')
    has_alt_loc_list = hasattr(data, 'alt_loc_list')
    has_sec_struct_list = hasattr(data, 'sec_struct_list')
    has_ins_code_list = hasattr(data, 'ins_code_list')
    has_sequence_index_list = hasattr(data, 'sequence_index_list')
    has_chain_name = hasattr(data, 'chain_name')

    chain_index = -1
    group_index = -1
    atom_index = -1

    atoms = [None] * len(data.x_coord_list)
    models = []

    for model_index in range(data.num_models):
        m = AtomicStructure(session, name=pdb_id)
        models.append(m)
        model_chain_count = data.chains_per_model[model_index]
        # traverse chains
        for _ in range(model_chain_count):
            chain_index += 1
            chain_id = data.chain_id_list[chain_index]
            if has_chain_name:
                chain_name = data.chain_name_list[chain_index]
            else:
                chain_name = chain_id
            chain_group_count = data.groups_per_chain[chain_index]

            # TODO:
            #  want 3-letter residue names, mmtf gives single-letter
            # m.set_input_seq_info(chain_name, seqres)
            # m.input_seq_source = "MMTF"

            # traverse groups
            last_ss = None
            ss_id = 0
            ins_code = b' '
            last_sequence_index = -1
            last_residue = None
            gap = False
            gaps = []
            for _ in range(chain_group_count):
                group_index += 1
                group_id = data.group_id_list[group_index]  # ordinal
                if has_ins_code_list:
                    ins_code = data.ins_code_list[group_index]
                    if ins_code == '\x00':
                        ins_code = ' '
                if has_sec_struct_list:
                    sec_struct = data.sec_struct_list[group_index]
                if has_sequence_index_list:
                    sequence_index = data.sequence_index_list[group_index]
                    gap = sequence_index != last_sequence_index + 1
                    last_sequence_index = sequence_index
                group_id = data.group_id_list[group_index]
                group_type = data.group_type_list[group_index]
                group = data.group_list[group_type]

                group_name = group['groupName']
                atom_name_list = group['atomNameList']
                element_list = group['elementList']
                bond_atom_list = group['bondAtomList']
                # formal_charge_list = group['formalChargeList']  # TODO
                # bond_order_list = group['bondOrderList']        # TODO

                r = m.new_residue(group_name, chain_name, group_id, ins_code)
                if gap:
                    gaps.append([last_residue, r])
                last_residue = r
                if has_sec_struct_list:
                    # Code    Name
                    #   0   pi helix
                    #   1   bend
                    #   2   alpha helix
                    #   3   extended
                    #   4   3-10 helix
                    #   5   bridge
                    #   6   turn
                    #   7   coil
                    #  -1   undefined
                    if sec_struct in {0, 2, 4}:
                        if last_ss is not HELIX:
                            ss_id += 1
                        r.is_helix = True
                        r.ss_id = ss_id
                        last_ss = HELIX
                    elif sec_struct == 3:
                        if last_ss is not STRAND:
                            ss_id += 1
                        r.is_sheet = True
                        r.ss_id = ss_id
                        last_ss = STRAND
                    else:
                        last_ss = None

                # traverse atoms
                start_atom = atom_index + 1
                group_alt_atoms = {}
                group_atom_count = len(atom_name_list)
                for i in range(group_atom_count):
                    atom_index += 1

                    x_coord = data.x_coord_list[atom_index]
                    y_coord = data.y_coord_list[atom_index]
                    z_coord = data.z_coord_list[atom_index]
                    if has_b_factor_list:
                        b_factor = data.b_factor_list[atom_index]
                    if has_occupancy_list:
                        occupancy = data.occupancy_list[atom_index]
                    if has_alt_loc_list:
                        alt_loc = data.alt_loc_list[atom_index]
                    atom_name = atom_name_list[i]
                    atom_element = element_list[i]
                    # formal_charge = formal_charge_list[i]        # TODO

                    if atom_name in group_alt_atoms:
                        a = group_alt_atoms[atom_name]
                        a.set_alt_loc(alt_loc, True)
                    else:
                        atoms[atom_index] = a = m.new_atom(atom_name, atom_element)
                        r.add_atom(a)
                        if has_alt_loc_list and alt_loc != '\x00':
                            a.set_alt_loc(alt_loc, True)
                            group_alt_atoms[atom_name] = a
                    a.coord = [x_coord, y_coord, z_coord]
                    if has_b_factor_list:
                        a.bfactor = b_factor
                    if has_occupancy_list:
                        a.occupancy = occupancy

                # connect bonds in residue
                for i in range(0, len(bond_atom_list), 2):
                    # bond_order = bond_order_list[i // 2]  # TODO
                    a0 = atoms[start_atom + bond_atom_list[i]]
                    a1 = atoms[start_atom + bond_atom_list[i + 1]]
                    if a0 is None or a1 is None:
                        # ignore bonds for alternate atoms
                        # assumes that all 'A' atoms were created first
                        continue
                    m.new_bond(a0, a1)
            # TODO:
            # create gap bonds
            # if gaps:
            #     pbg = PseudoBondGroup('missing structure')
            #     pdb = session.pb_manager.get_group('missing structure')
            #     for r0, r1 in gaps:
            #         connect_closest(r0, r1)

    if has_bond_atom_list:
        # traverse inter-group bonds
        xlinks = None
        bond_atom_list = data.bond_atom_list
        for i in range(0, len(bond_atom_list), 2):
            # bond_order = data.bond_order_list[i // 2]    # TODO
            a0 = atoms[bond_atom_list[i]]
            a1 = atoms[bond_atom_list[i + 1]]
            # TODO:
            #    can bonds connect two models?
            #    if so, create pseudobond
            m0 = a0.structure
            m1 = a1.structure
            if m0 == m1:
                m0.new_bond(a0, a1)
            # else:
            #     if xlinks is None:
            #         xlinks = session.pb_manager.get_group('cross links')
            #     xlinks.new_pseudobond(a0, a1)

    # TODO:
    # for m in models:
    #     find_and_add_metal_coordination_bonds(m)

    return models, ("Opened MMTF data containing %d atoms and %d bonds"
                    % (sum(m.num_atoms for m in models),
                       sum(m.num_bonds for m in models)))
