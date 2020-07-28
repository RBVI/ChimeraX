..  vim: set expandtab shiftwidth=4 softtabstop=4:

..
    === UCSF ChimeraX Copyright ===
    Copyright 2018 Regents of the University of California.
    All rights reserved.  This software provided pursuant to a
    license agreement containing restrictions on its disclosure,
    duplication and use.  For details see:
    http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html
    This notice must be embedded in or attached to all copies,
    including partial copies, of the software or any revisions
    or derivations thereof.
    === UCSF ChimeraX Copyright ===

===============================
Writing mmCIF Files in ChimeraX
===============================

:Date: April 2018

.. _Greg Couch: mailto:gregc@cgl.ucsf.edu
.. _Resource for Biocomputing, Visualization, and Informatics: http://www.rbvi.ucsf.edu/

.. |---| unicode:: U+2014  .. em dash

------------
Introduction
------------

There are several goals for ChimeraX's mmCIF writer:
   
  * write out the information that ChimeraX wants to read

  * output inferred parts of mmCIF (so other mmCIF readers don't have to infer the same things)

  * use stylized output (for readability and fast reading)

  * output should pass the PDB's online deposition validator

  * output should pass the PDB's mmCIF validation software

  * full connectivity should optionally be generated (not done)

-------------------
What ChimeraX Wants
-------------------

What ChimeraX wants to read from a mmCIF file is documented in
:doc:`ChimeraX Fast mmCIF Guidelines <mmcif_guidelines>`.
Saving connectivity is a major issue and is discussed separately.

---------------------
Inferred mmCIF Tables
---------------------

Many data relationships in a mmCIF file can be inferred.
For example, from the contents of the **atom_site** table,
the chain identifier to entity identifier mapping can be computed
(the **struct_asym** table).
Consequently, ChimeraX's mmCIF reader skips reading tables when it can
infer all of the information it needs from a table.
Other application's mmCIF readers might not compute the same information,
so ChimeraX outputs the tables it infers for completeness.
This is also needed for the mmCIF files to validate.

---------------
Stylized Output
---------------

As shown in :doc:`Benchmarking readcif <../mmcif_cpp/readcif_cpp/docs/compare>`,
stylized PDBx/mmCIF output can be read faster than unstylized output.
It is also easier to visually scan fixed column width tables for
interesting values.
ChimeraX ouputs the **chimerax_audit_syntax.case_sensitive_flag** as **Y**
to indicate that all keywords are lowercase and appear at the beginning
of a line.
And outputs **chimerax_audit_syntax.fixed_width** with just the
**atom_site** and **atom_site_anisotrop** tables listed
(since the those tables are typically the largest ones in the mmCIF file).

----------
Validation
----------

We tested the mmCIF output in two ways:

  * Using the `World Wide PDB`_'s `online validator`

  * Validated the mmCIF file using the associated mmCIF dictionary

There are several software packages that will try to validate a mmCIF file
using the associate dictionary.
We used the `mmCIF Dictionary Suite`_ from the wwPDB,
since it supports the current mmCIF dictionary (version 5).

.. _World Wide PDB: https://www.wwpdb.org/
.. _online validator: https://www.wwpdb.org/deposition/preparing-pdbx-mmcif-files
.. _mmCIF Dictionary Suite: https://sw-tools.rcsb.org/apps/MMCIF-DICT-SUITE/index.html

------------
Connectivity
------------

TODO: full connectivity should optionally be generated

--------
Problems
--------

ChimeraX does not save enough information to completely regenerate
some of the mmCIF tables it uses.

Heterogeneous information is discarded when reading,
so it is not present when writing.

ChimeraX is only concerned about strands, so
the sheet information in the **struct_sheet_range** table is lost.
On output, the sheet identifier is given as unknown (**?**).

In other cases, the original mmCIF table, that is copied verbatim into the output,
is non-conforming.
For example, in wwPDB provided mmCIF files,
the *mandatory* item **pdbx_src_id** is often missing
from the **entity_src_gen** and **entity_src_nat** tables.

Only the single letter code for a residue in a chain's sequence is kept.
So, if that residue is not present in any entity with the same sequence,
then the name of the residue defaults to the standard one for that letter.

----------------------------------------
Generated mmCIF Categories and Keywords
----------------------------------------

For reference,
all of the mmCIF categories and keywords that ChimeraX outputs are listed below.

Recognized Data Categories and Keywords
---------------------------------------

   +----------------------------+----------------------------------------+
   |      Category              | Keywords                               |
   +============================+========================================+
   | atom_type                  |                                        |
   |                            | symbol                                 |
   +----------------------------+----------------------------------------+
   | atom_site                  |                                        |
   |                            | group_PDB, id, label_entity_id,        |
   |                            | label_asym_id, auth_asym_id,           |
   |                            | pdbx_PDB_ins_code, label_seq_id,       |
   |                            | auth_seq_id, label_alt_id,             |
   |                            | type_symbol, label_atom_id,            |
   |                            | label_comp_id, Cartn_x, Cartn_y,       |
   |                            | Cartn_z, occupancy, B_iso_or_equiv,    |
   |                            | pdbx_PDB_model_num                     |
   +----------------------------+----------------------------------------+
   | atom_site_anisotrop        |                                        |
   |                            | id, U[1]_[1], U[1]_[2], U[1]_[3],      |
   |                            | U[2]_[2], U[2]_[3], U[3]_[3]           |
   +----------------------------+----------------------------------------+
   | audit_conform              |                                        |
   |                            | dict_name, dict_version                |
   +----------------------------+----------------------------------------+
   | chimerax_audit_syntax      |                                        |
   |                            | case_sensitive_flags, fixed_width      |
   +----------------------------+----------------------------------------+
   | cell                       |                                        |
   |                            | *copied from original file*            |
   +----------------------------+----------------------------------------+
   | chem_comp                  |                                        |
   |                            | id, type,                              |
   |                            | **name** *extract from original file*  |
   +----------------------------+----------------------------------------+
   | citation                   |                                        |
   |                            | *merged from original file*            |
   |                            | id, title, journal_abbrev,             |
   |                            | journal_volume, year, page_first,      |
   |                            | page_last, journal_issue,              |
   |                            | pdbx_database_id_PubMed,               |
   |                            | pdbx_database_id_DOI                   |
   +----------------------------+----------------------------------------+
   | citation_author            |                                        |
   |                            | citation_id, name, ordinal             |
   +----------------------------+----------------------------------------+
   | entry                      |                                        |
   |                            | id                                     |
   +----------------------------+----------------------------------------+
   | entity                     |                                        |
   |                            | id, type, pdbx_description             |
   +----------------------------+----------------------------------------+
   | entity_poly                |                                        |
   |                            | entity_id, type, nstd_monomer,         |
   |                            | pdbx_seq_one_letter_code_can           |
   +----------------------------+----------------------------------------+
   | entity_poly_seq            |                                        |
   |                            | entity_id, num, mon_id                 |
   +----------------------------+----------------------------------------+
   | entity_src_gen             |                                        |
   |                            | *copied from original file*            |
   +----------------------------+----------------------------------------+
   | entity_src_nat             |                                        |
   |                            | *copied from original file*            |
   +----------------------------+----------------------------------------+
   | pdbx_poly_seq_scheme       |                                        |
   |                            | entity_id, asym_id, mon_id, seq_id,    |
   |                            | pdb_strand_id, pdb_seq_num,            |
   |                            | pdb_ins_code                           |
   +----------------------------+----------------------------------------+
   | pdbx_struct_assembly       |                                        |
   |                            | *copied from original file*            |
   +----------------------------+----------------------------------------+
   | pdbx_struct_assembly_gen   |                                        |
   |                            | *copied from original file*            |
   +----------------------------+----------------------------------------+
   | pdbx_struct_oper_list      |                                        |
   |                            | *copied from original file*            |
   +----------------------------+----------------------------------------+
   | software                   |                                        |
   |                            | name, version, location,               |
   |                            | classification, os, type, citation_id, |
   |                            | pdbx_ordinal                           |
   +----------------------------+----------------------------------------+
   | struct_asym                |                                        |
   |                            | id, entity_id                          |
   +----------------------------+----------------------------------------+
   | struct_conf                |                                        |
   |                            | id, conf_type_id, beg_label_asym_id,   |
   |                            | beg_label_comp_id, beg_label_seq_id,   |
   |                            | end_label_asym_id, end_label_comp_id,  |
   |                            | end_label_seq_id,                      |
   |                            | beg_auth_asym_id, beg_auth_seq_id,     |
   |                            | pdbx_beg_PDB_ins_code,                 |
   |                            | end_auth_asym_id, end_auth_seq_id,     |
   |                            | pdbx_end_PDB_ins_code,                 |
   +----------------------------+----------------------------------------+
   | struct_conf_type           |                                        |
   |                            | id                                     |
   +----------------------------+----------------------------------------+
   | struct_conn                |                                        |
   |                            | id, conn_type_id,                      |
   |                            | ptnr1_label_asym_id,                   |
   |                            | ptnr1_auth_asym_id,                    |
   |                            | pdbx_ptnr1_PDB_ins_code,               |
   |                            | ptnr1_label_seq_id,                    |
   |                            | ptnr1_auth_seq_id,                     |
   |                            | pdbx_ptnr1_label_alt_id,               |
   |                            | ptnr1_label_atom_id,                   |
   |                            | ptnr1_label_comp_id,                   |
   |                            | ptnr1_symmetry,                        |
   |                            | ptnr2_label_asym_id,                   |
   |                            | ptnr2_auth_asym_id,                    |
   |                            | pdbx_ptnr2_PDB_ins_code,               |
   |                            | ptnr2_label_seq_id,                    |
   |                            | ptnr2_auth_seq_id,                     |
   |                            | pdbx_ptnr2_label_alt_id,               |
   |                            | ptnr2_label_atom_id,                   |
   |                            | ptnr2_label_comp_id,                   |
   |                            | ptnr2_symmetry, pdbx_dist_value        |
   +----------------------------+----------------------------------------+
   | struct_conn_type           |                                        |
   |                            | id                                     |
   +----------------------------+----------------------------------------+
   | struct_sheet_range         |                                        |
   |                            | sheet_id, id,                          |
   |                            | beg_label_asym_id, beg_label_comp_id,  |
   |                            | beg_label_seq_id, end_label_asym_id,   |
   |                            | end_label_comp_id, end_label_seq_id    |
   |                            | symmetry, beg_auth_asym_id,            |
   |                            | beg_auth_seq_id,                       |
   |                            | pdbx_beg_PDB_ins_code,                 |
   |                            | end_auth_asm_id, end_auth_seq_id,      |
   |                            | pdbx_end_PDB_ins_code                  |
   +----------------------------+----------------------------------------+
   | symmetry                   |                                        |
   |                            | *copied from original file*            |
   +----------------------------+----------------------------------------+
