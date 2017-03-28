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

    audit_conform
        Tells which CIF dicitionary the data conforms too.
        Should be required in every CIF file to disambiguate
        how the data should be interpreted.
        Currently only present in wwPDB mmCIF files,
        so it used in heuristic to speed up CIF parsing.

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

    struct_conn
        Contains non-standard connectivity.
        Standard amino and nucleic acid connectivity is given by chemical
        component templates.

    struct_conf
        Contains helix and turn residue ranges.
        Formerly held strand residue ranges
        but that information is now in the *struct_sheet_range*
        data.

    struct_sheet_range
        Contains strand residue ranges and associated sheets.

    entity_poly_seq
        Contains the sequence of residues in a chain.
        Used to know which residues to connect and where there are structural gaps.

    entry
        Contains the 4-letter PDB identifier.
        Used to tell user if there is a newer version available.

    pdbx_database_PDB_obs_spr
        Contains information about obsolete and superseded PDB entries.
        Used to tell user if there is a newer version available.

    chem_comp
        Contains information about the chemical components in the structure.
        Used for local residue templates.

    chem_comp_bond
        Contains connectivity of chemical components.
        Used for local residue templates.
        Currently only present in "udpated" PDB files from the PDBe.
        Metal coordination bonds are incorrectly given as covalent bonds,
        so the residues needs to be postprocessed.

    pdbx_struct_assembly
        Contains information "about the structural elements that form
        macromolecular assemblies."

    pdbx_struct_assembly_gen
        Contains information "about the generation of each
        macromolecular assemblies."

    entity
        Contains details "about the molecular entities that are
        present in the crystallographic structure."
        Used to extract description of chains.

    entity_src_gen
        Contains "details of the source from wich the entity was obtained
        in cases where the source was genetically manipulated."
        Used to extract scientific name of entities.

    entity_src_nat
        Contains "details of the source from wich the entity was obtained
        in cases where the entity was isolated directly from a natural tissue."
        Used to extract scientific name of entities.
