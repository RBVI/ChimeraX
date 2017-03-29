..  vim: set expandtab shiftwidth=4 softtabstop=4:

.. 
    === UCSF ChimeraX Copyright ===
    Copyright 2017 Regents of the University of California.
    All rights reserved.  This software provided pursuant to a
    license agreement containing restrictions on its disclosure,
    duplication and use.  For details see:
    http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html
    This notice must be embedded in or attached to all copies,
    including partial copies, of the software or any revisions
    or derivations thereof.
    === UCSF ChimeraX Copyright ===

mmcif: mmCIF input/output
=========================

.. automodule:: chimerax.core.atomic.mmcif
    :members:
    :show-inheritance:

Use of mmCIF
~~~~~~~~~~~~

    ChimeraX does not read all the mmCIF data.  It only reads
    what it needs to correctly reconstruct the structure,
    and provide information about the structure.

    The following is a list of the recognized mmCIF category names
    with a brief description of the data they contain and how they are used.
    To minimize the about of data read (and the associated parsing cost),
    ChimeraX takes advantage of the cross-referencing of data
    within the mmCIF file,
    that allows the reconstruction of data relationships
    without reading the table that explicitly lists the relationship.

    atom_site
        Contains atom coordinates.
        Typically the largest table in the mmCIF file.
        wwPDB files used fixed width columns for the data,
        and that knowledge is used to speed up parsing.

    atom_site_anisotrop
        Contains anisotropic displacement data for atoms.
        While this data could be combined with the *atom_site* data,
        in practice it is not.
        Consequently, ChimeraX only looks in the *atom_site_anisotrop*
        table for the anisotropic displacement data.
        wwPDB files used fixed width columns for the data,
        and that knowledge is used to speed up parsing.

    audit_conform
        Tells which CIF dicitionary the data conforms too.
        Should be required in every CIF file to disambiguate
        how the data should be interpreted.
        Currently only present in wwPDB mmCIF files,
        so it used in heuristic to speed up CIF parsing.

    chem_comp
        Contains information about the chemical components in the structure.
        Used for local residue templates.

    chem_comp_bond
        Contains connectivity of chemical components.
        Used for local residue templates.
        Currently only present in "udpated" PDB files from the PDBe.
        Metal coordination bonds are incorrectly given as covalent bonds,
        so the residues needs to be postprocessed.

    entity
        Contains details "about the molecular entities that are
        present in the crystallographic structure."
        Used to extract description of chains.

    entity_poly_seq
        Contains the sequence of residues in a chain.
        Used to know which residues to connect and where there are structural gaps.

    entity_src_gen
        Contains "details of the source from wich the entity was obtained
        in cases where the source was genetically manipulated."
        Used to extract scientific name of entities.

    entity_src_nat
        Contains "details of the source from wich the entity was obtained
        in cases where the entity was isolated directly from a natural tissue."
        Used to extract scientific name of entities.

    entry
        Contains the 4-letter PDB identifier.
        Used to tell user if there is a newer version available.

    pdbx_database_PDB_obs_spr
        Contains information about obsolete and superseded PDB entries.
        Used to tell user if there is a newer version available.

    pdbx_struct_assembly
        Contains information "about the structural elements that form
        macromolecular assemblies."

    pdbx_struct_assembly_gen
        Contains information "about the generation of each
        macromolecular assemblies."

    pdbx_struct_oper_list
        Contains transform matrix for symmetry operations.

    struct_conf
        Contains helix and turn residue ranges.
        Formerly held strand residue ranges
        but that information is now in the *struct_sheet_range*
        data.

    struct_conn
        Contains non-standard connectivity.
        Standard amino and nucleic acid connectivity is given by chemical
        component templates.

    struct_sheet_range
        Contains strand residue ranges and associated sheets.

    .. |req| †

    +---------------------------+----------------------------+
    |      Category              | Keywords (|req| required) |
    +----------------------------+---------------------------+
    | atom_site                  | id                        |
    |                            +---------------------------+
    |                            | label_entity_id           |
    |                            +---------------------------+
    |                            | label_asym_id|req|        |
    |                            +---------------------------+
    |                            | auth_asym_id              |
    |                            +---------------------------+
    |                            | pdbx_PDB_ins_code         |
    |                            +---------------------------+
    |                            | label_seq_id|req|         |
    |                            +---------------------------+
    |                            | auth_seq_id               |
    |                            +---------------------------+
    |                            | label_alt_id              |
    |                            +---------------------------+
    |                            | type_symbol|req|          |
    |                            +---------------------------+
    |                            | label_atom_id|req|        |
    |                            +---------------------------+
    |                            | auth_atom_id              |
    |                            +---------------------------+
    |                            | label_comp_id|req|        |
    |                            +---------------------------+
    |                            | auth_comp_id              |
    |                            +---------------------------+
    |                            | Cartn_x|req|              |
    |                            +---------------------------+
    |                            | Cartn_y|req|              |
    |                            +---------------------------+
    |                            | Cartn_z|req|              |
    |                            +---------------------------+
    |                            | occupancy                 |
    |                            +---------------------------+
    |                            | B_iso_or_equiv            |
    |                            +---------------------------+
    |                            | pdbx_PDB_model_num        |
    +----------------------------+---------------------------+
    | atom_site_anisotrop        | id|req|                   |
    |                            +---------------------------+
    |                            | U[1]_[1]|req|             |
    |                            +---------------------------+
    |                            | U[1]_[2]|req|             |
    |                            +---------------------------+
    |                            | U[1]_[3]|req|             |
    |                            +---------------------------+
    |                            | U[2]_[2]|req|             |
    |                            +---------------------------+
    |                            | U[2]_[3]|req|             |
    |                            +---------------------------+
    |                            | U[3]_[3]|req|             |
    +----------------------------+---------------------------+
    | audit_conform              | dict_name                 |
    |                            +---------------------------+
    |                            | dict_version              |
    |                            +---------------------------+
    |                            | pdbx_keywords             |
    |                            +---------------------------+
    |                            | pdbx_fixed_width_columns  |
    +----------------------------+---------------------------+
    | chem_comp                  | id|req|                   |
    |                            +---------------------------+
    |                            | type|req|                 |
    +----------------------------+---------------------------+
    | chem_comp_bond             | comp_id|req|              |
    |                            +---------------------------+
    |                            | atom_id_1|req|            |
    |                            +---------------------------+
    |                            | atom_id_2|req|            |
    +----------------------------+---------------------------+
    | entity_poly_seq            | entity_id|req|            |
    |                            +---------------------------+
    |                            | num|req|                  |
    |                            +---------------------------+
    |                            | mon_id|req|               |
    |                            +---------------------------+
    |                            | hetero                    |
    +----------------------------+---------------------------+
    | entity                     | id|req|                   |
    |                            +---------------------------+
    |                            | pdbx_description          |
    +----------------------------+---------------------------+
    | entity_src_gen             | entity_id|req|            |
    |                            +---------------------------+
    |                            | pdbx_gene_src_scientific_name|req| |
    +----------------------------+---------------------------+
    | entity_src_nat             | entity_id|req|            |
    |                            +---------------------------+
    |                            | pdbx_organism_scientific|req| |
    +----------------------------+---------------------------+
    | entry                      | id|req|                   |
    +----------------------------+---------------------------+
    | pdbx_database_PDB_obs_spr  | id|req|                   |
    |                            +---------------------------+
    |                            + pdb_id|req|               |
    |                            +---------------------------+
    |                            + replace_pdb_id|req|       |
    +----------------------------+---------------------------+
    | pdbx_struct_assembly       | id|req|                   |
    |                            +---------------------------+
    |                            | details|req|              |
    +----------------------------+---------------------------+
    | pdbx_struct_assembly_gen   | assembly_id|req|          |
    |                            +---------------------------+
    |                            | oper_expression|req|      |
    |                            +---------------------------+
    |                            | asym_id_list|req|         |
    +----------------------------+---------------------------+
    | pdbx_struct_oper_list      | id|req|                   |
    |                            +---------------------------+
    |                            | matrix[1][1]|req|         |
    |                            +---------------------------+
    |                            | matrix[1][2]|req|         |
    |                            +---------------------------+
    |                            | matrix[1][3]|req|         |
    |                            +---------------------------+
    |                            | matrix[2][1]|req|         |
    |                            +---------------------------+
    |                            | matrix[2][2]|req|         |
    |                            +---------------------------+
    |                            | matrix[2][3]|req|         |
    |                            +---------------------------+
    |                            | matrix[3][1]|req|         |
    |                            +---------------------------+
    |                            | matrix[3][2]|req|         |
    |                            +---------------------------+
    |                            | matrix[3][3]|req|         |
    |                            +---------------------------+
    |                            | vector[1]|req|            |
    |                            +---------------------------+
    |                            | vector[2]|req|            |
    |                            +---------------------------+
    |                            | vector[3]|req|            |
    +----------------------------+---------------------------+
    | struct_conf                | id|req|                   |
    |                            +---------------------------+
    |                            | conf_type_id|req|         |
    |                            +---------------------------+
    |                            | beg_label_asym_id|req|    |
    |                            +---------------------------+
    |                            | beg_label_comp_id|req|    |
    |                            +---------------------------+
    |                            | beg_label_seq_id|req|     |
    |                            +---------------------------+
    |                            | end_label_asym_id|req|    |
    |                            +---------------------------+
    |                            | end_label_comp_id|req|    |
    |                            +---------------------------+
    |                            | end_label_seq_id|req|     |
    +----------------------------+---------------------------+
    | struct_conn                | conn_type_id|req|         |
    |                            +---------------------------+
    |                            | ptnr1_label_asym_id|req|  |
    |                            +---------------------------+
    |                            | pdbx_ptnr1_PDB_ins_code   |
    |                            +---------------------------+
    |                            | ptnr1_label_seq_id|req|   |
    |                            +---------------------------+
    |                            | ptnr1_auth_seq_id         |
    |                            +---------------------------+
    |                            | pdbx_ptnr1_label_alt_id   |
    |                            +---------------------------+
    |                            | ptnr1_label_atom_id|req|  |
    |                            +---------------------------+
    |                            | ptnr1_label_comp_id|req|  |
    |                            +---------------------------+
    |                            | ptnr1_symmetry            |
    |                            +---------------------------+
    |                            | ptnr2_label_asym_id|req|  |
    |                            +---------------------------+
    |                            | pdbx_ptnr2 _PDB_ins_code  |
    |                            +---------------------------+
    |                            | ptnr2_label_seq_id|req|   |
    |                            +---------------------------+
    |                            | ptnr2_auth_seq_id         |
    |                            +---------------------------+
    |                            | pdbx_ptnr2 _label_alt_id  |
    |                            +---------------------------+
    |                            | ptnr2_label_atom_id|req|  |
    |                            +---------------------------+
    |                            | ptnr2_label_comp_id|req|  |
    |                            +---------------------------+
    |                            | ptnr2_symmetry            |
    |                            +---------------------------+
    |                            | pdbx_dist_value           |
    +------------------------------+-------------------------+
    | struct_sheet_range         | sheet_id|req|             |
    |                            +---------------------------+
    |                            | id|req|                   |
    |                            +---------------------------+
    |                            | beg_label_asym_id|req|    |
    |                            +---------------------------+
    |                            | beg_label_comp_id|req|    |
    |                            +---------------------------+
    |                            | beg_label_seq_id|req|     |
    |                            +---------------------------+
    |                            | end_label_asym_id|req|    |
    |                            +---------------------------+
    |                            | end_label_comp_id|req|    |
    |                            +---------------------------+
    |                            | end_label_seq_id|req|     |
    +----------------------------+---------------------------+
