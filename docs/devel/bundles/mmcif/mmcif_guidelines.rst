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

==============================
ChimeraX Fast mmCIF Guidelines
==============================

:Date: April 2017

.. _Greg Couch: mailto:gregc@cgl.ucsf.edu
.. _Resource for Biocomputing, Visualization, and Informatics: http://www.rbvi.ucsf.edu/

.. |---| unicode:: U+2014  .. em dash

------------
Introduction
------------

ChimeraX's mmCIF reader is fast.
It is fast because:

  * it uses the `readcif` C++ library to parse a mmCIF file

  * it takes advantage of the wwPDB's PDBx/mmCIF styling

  * it only fully parses what it needs

  * it doesn't create Python objects for each atom, bond, residue, nor chain

Being fast is good, but accurately reconstructing atomic structures is better.
ChimeraX does the best it can with the data it is given.
Some errors are detected and logged,
but the ultimate responsibility for the fidelity of the data
belongs to the user.

The following discusses what is needed to be fast,
and what data is understood, by ChimeraX.

.. _readcif: https://github.com/RBVI/readcif

-----------------
mmCIF Terminology
-----------------

.. _mmCIF: https://mmcif.wwpdb.org/
.. _CIF: https://www.iucr.org/resources/cif

`mmCIF`_,
MacroMolecular Crystallographic Interchange Format, is a file format
for describing large atomic structures and is
based on the `CIF`_ format
for small molecules.
The file content is human-readable
and is organized as a series of categories, formatted as tables.
A table consists of a series of column names followed
by one or more rows of data.
The column names are period separated category name followed by a keyword,
*e.g.*, **audit_conform.dict_name**.
A table with a single row can be written as a series of pairs of column name and data value.
The categories, keywords, and the domain and range of the data
are defined in a dictionary written in the CIF format.
Using the dictionary, an mmCIF file can be validated (just like
an XML file with its corresponding XML Schema).

-----------------------
Which mmCIF dictionary?
-----------------------

A generic problem with CIF files is that
they usually don't embed which dictionary the file corresponds to.
This is exacerbated by the fact
that semantically different file types share the same **.cif** file suffix,
and that the dictionaries for the mmCIF and CIF formats overlap
but do not describe atomic structures in the same way.
Consequently, when given a **.cif** file, users cannot predict
whether the file is for small molecules or macromolecules.
Furthermore, applications cannot reliably parse
**.cif** files and may need to query the user for guidance.
A simple solution would be to use unique suffixes for different
type of CIF files, *e.g.* use **.cif** for small
molecules but **.mmcif** for macromolecules,
but this is not current practice.

.. _World Wide PDB: http://www.wwpdb.org/
.. _PDB Europe: http://www.pdbe.org/
.. _updated mmCIF files: http://europepmc.org/abstract/MED/26476444
.. _PDBx/mmCIF dictionary: http://mmcif.wwpdb.org/

The `World Wide PDB`_
addresses the missing dictionary information by
explicitly listing the data definition dictionary name
and version in the **audit_conform** table in
mmCIF files it distributes.
Unfortunately,
`PDB Europe`_'s `updated mmCIF files`_,
currently do not include this information,
thus making it difficult to validate these files.
ChimeraX expects **.cif** files to conform to a published
`PDBx/mmCIF dictionary`_,
whether the data definition dictionary information is present or not.

-----------
Performance
-----------

Using a data definition dictionary to guide the parsing of a CIF file,
while flexible, tends to be slow because interpreting each datum
requires looking up its definition it the dictionary.
Instead, for speed, ChimeraX hardcodes information
for a few select categories,
skips over categories it doesn't need,
and treats other categories as generic tables of string data.

The ChimeraX code for reading CIF files consists of two parts:

  * a :doc:`fast CIF parser <../mmcif_cpp/readcif_cpp/docs/index>`
    that provides the framework for fast parsing
    (skipping unused categories, PDBx/mmCIF styling), and

  * a Python module (implemented in C++) that converts
    the parsed data into ChimeraX's internal data structures.

ChimeraX takes advantage of the cross-referencing of data
within a mmCIF file to reconstruct data relationships
without reading the tables that explicitly list those relationships.
For example, the mapping of author identifiers to normative identifiers
is contained in both the **pdbx_poly_seq_scheme** table
and the **atom_site** table.
Since the **atom_site** table is always parsed,
we were able to speed up reading a mmCIF file by about 3%
by retaining the information from the **atom_site** table and
not parsing the **pdbx_poly_seq_scheme** table at all.

Stylized PDBx/mmCIF Files
-------------------------

.. _styling: http://mmcif.wwpdb.org/docs/faqs/pdbx-mmcif-faq-general.html">styling
.. _mmcif_pdbx dictionary: http://mmcif.wwpdb.org/pdbx-mmcif-home-page.html

mmCIF files from the World-Wide PDB (wwPDB) are typically formatted for fast parsing.
This is known as PDBx/mmCIF
`styling`.
If a CIF file is known to use PDBx/mmCIF stylized formatting,
then parsing can be almost :doc:`four times faster <../mmcif_cpp/readcif_cpp/docs/compare>`.
Currently, ChimeraX uses a heuristic to detect that a mmCIF file is stylized:
it is assumed only when a mmCIF file uses the
`mmcif_pdbx dictionary`_ version 4 or later.
However, it is preferrable to explicity enable fast stylized parsing by setting the values
of specific annotation flags in the CIF file.
ChimeraX has added metadata in **audit_syntax** category
with explicit annotations as detailed below.
(In the future ChimeraX's use of a heuristic may be discontinued after explicit annotations becomes widespread.)

The important aspects of styling are:
(1) reserved words and tags are always case-sensitive, and
(2) categories with fixed column width tables are explicitly so noted.

.. _mmcif_pdbx.dic: http://mmcif.wwpdb.org/dictionaries/mmcif_pdbx.dic/Index/

    Case-sensitive words and tags (conformance should be explicitly annotated with
    ``audit_syntax.case_sensitive_flag Y``\ ):

      * CIF reserved words *must* be in lowercase

      * Category names and keywords *must* match the case given in the associated dictionary,
        *e.g.*, **atom_site.Cartn_x** in `mmcif_pdbx.dic`_

      * CIF reserved words and tags *must* only appear immediately after an ASCII newline

    Fixed width column tables (conformance should be explicitly annotated with
    ``audit_syntax.fixed_width`` followed by a space separated list of categories):

      * All columns *must* be left-aligned

      * Each row of data *must* include all columns

      * All rows *must* be the same length, using trailing spaces as padding

      * The end of the category's data values *is* terminated by a comment line

    Performance improvements are especially noticeable when processing large tables such as
    **atom_site** and **atom_site_ansitrop**.

---------------------------
Reconstructing Connectivity
---------------------------

One of the deficiencies of the mmCIF documentation
is the lack of a published protocol for reconstructing the atomic connectivity.

The connectivity between residues is not given for standard amino and nucleic acids.
Rather, it is inferred from the polymer sequence data.

.. _Chemical Component Dictionary: http://wwpdb.org/data/ccd
.. _RCSB PDB: http://www.rcsb.org/
.. _Ligand Expo: http://ligand-expo.rcsb.org/

The internal connectivity of residues is not given in the wwPDB's mmCIF files.
That information is available separately in a
`Chemical Component Dictionary`_, CCD,
that "is updated with each weekly PDB release."
ChimeraX uses the Internet to fetch individual residue templates from the
`RCSB PDB`_'s `Ligand Expo`_
instead of having users update the huge CCD each week.
However, there are at least two curation problems with the residue templates:
(1) the templates are sometimes incomplete, *e.g.*,
missing the H1 and H3 for amino acids at the N-terminus of proteins
(the UNL and UNX templates intentionally have no atoms nor bonds because there is no implied connectivity),
and (2) the templates sometimes incorrectly identify metal coordination bonds as covalent bonds (*e.g.*, HEM).
In both cases, custom code has to be written to correct the problem.
(In the case of (1) above, the wwPDB has alternate templates with
protonation variants for the standard amino acids.
But the general case requires that bonds be computed using element-based
distance cutoffs.)

Another potential problem arises when a residue template is not available, *e.g.*,
a mmCIF file of a new structure not yet deposited in the PDB.
In this case, a template should be embedded directly in the mmCIF file with
the **chem_comp** and **chem_comp_bond** tables.
As a last resort, if a template is missing or incomplete,
ChimeraX will connect the residue using element-dependent bond distances
|---| *ideally* this should never be necessary.

Finally, the treatment of waters in wwPDB mmCIF files potentially presents a problem.
The **atom_site**'s **label_comp_id**, **label_asym_id**, **label_entity_id**, and **label_seq_id** data values are identical,
so the waters appear to be all in one residue.
(If they were unique, they could be used, along with the other **label_** keywords,
as a unique key for a database table.)
Fortunately, in practice, the optional **auth_seq_id** keyword's data values are usually
included in the file and can be used to distinguish each water.
Any mmCIF files without unique **auth_seq_id**\ s
must have unique **label_seq_id**\ s,
that is, the solvent *must* be uniquely numbered to indicate that the residues are distinct.

With the above considerations, the connectivity protocol becomes,
for each CIF data block:

  #. Read **audit_conform** and/or **audit_syntax** for metadata needed to speed up parsing

  #. Read **chem_comp** and **chem_comp_bond** for embedded residue templates

  #. Read **entity_poly_seq** for sequence information (and thus polymer connectivity)

  #. Read **atom_site** for atomic coordinates

  #. Read **struct_conn** for non-standard connectivity

  #. Assemble the atomic structure while compensating for the above deficiencies.

Multiple CIF data blocks are treated as multiple atomic structures.

Embedded Residue Templates
--------------------------

The PDBe's updated mmCIF files embed residue templates for connectivity.
This means that the **chem_comp_bond** and **chem_comp_atom** tables
for all residue types in the structure are added to the mmCIF file.
A reasonable method for creating the **chem_comp_bond**
the **chem_comp_atom** tables
is to concatenate the corresponding tables from the various CCD residue
templates listed in the **chem_comp** table.
Including these two tables makes the mmCIF files self-contained,
*i.e.*, no templates need to be fetched via the Internet.

--------------
Best Practices
--------------

ChimeraX performs a linear scan of a mmCIF file for the data it needs.
To avoid the memory cost of saving information before it is needed,
ChimeraX will note where a category's data is in the file
and then backtrack to parse that data when it's needed.
Re-reading data takes time,
so having the data in the desired order can speed up processing a file considerably.

The best presentation order of the mmCIF data for ChimeraX is as follows:

  1. **audit_syntax** table near beginning of the file and:

    a) explicitly give PDBx/mmCIF styling information (*e.g.*,
       that the **atom_site** table uses fixed width columns)

  2. Connectivity information for non-standard residues, with
     the **chem_comp** table preceding the **chem_comp_bond** table
  3. **entity_poly_seq** table (sequence information)
  4. **atom_site** table (coordinate data)
  5. **atom_site_anisotrop** table
  6. **struct_conn** table
  7. **struct_conf** table
  8. **struct_sheet_range** table

The order in which other tables appear does not currently matter.
For future compatibility be sure to define data before it is referenced.
For example, the **entity** table should come before the **entity_poly_seq** table.

----------------------------------------
Recognized mmCIF Categories and Keywords
----------------------------------------

For reference,
all of the mmCIF categories and keywords that ChimeraX parses are listed below.
Some keywords are required to be present in a category for its data to be used.
Afterwards,
there is a brief description of the categories and why they are important.
All of the categories are considered optional,
but if one is missing,
then ChimeraX might incorrectly infer what could have been explicitly given.
For instance, if the tables for the secondary structure categories are missing
then ChimeraX needs to compute that information.
Also, the **atom_site** table is effectively required
because, without it, there is no resulting atomic structure.

.. |req| unicode:: U+2020 .. dagger
   :ltrim:

Recognized Data Categories and Keywords
---------------------------------------

   +----------------------------+----------------------------------------+
   |      Category              | Keywords (|req| = required)            |
   +============================+========================================+
   | atom_site                  |                                        |
   |                            | id, label_entity_id,                   |
   |                            | label_asym_id |req|, auth_asym_id,     |
   |                            | pdbx_PDB_ins_code, label_seq_id |req|, |
   |                            | auth_seq_id, label_alt_id,             |
   |                            | type_symbol |req|, label_atom_id |req|,|
   |                            | auth_atom_id, label_comp_id |req|,     |
   |                            | auth_comp_id, Cartn_x |req|,           |
   |                            | Cartn_y |req|, Cartn_z |req|,          |
   |                            | occupancy, B_iso_or_equiv,             |
   |                            | pdbx_PDB_model_num                     |
   +----------------------------+----------------------------------------+
   | atom_site_anisotrop        |                                        |
   |                            | id |req|, U[1]_[1] |req|,              |
   |                            | U[1]_[2] |req|, U[1]_[3] |req|,        |
   |                            | U[2]_[2] |req|, U[2]_[3] |req|,        |
   |                            | U[3]_[3] |req|                         |
   +----------------------------+----------------------------------------+
   | audit_conform              |                                        |
   |                            | dict_name, dict_version                |
   +----------------------------+----------------------------------------+
   | audit_syntax               |                                        |
   |                            | case_sensitive_flag, fixed_width       |
   +----------------------------+----------------------------------------+
   | chem_comp                  |                                        |
   |                            | id |req|, type |req|                   |
   +----------------------------+----------------------------------------+
   | chem_comp_bond             |                                        |
   |                            | comp_id |req|, atom_id_1 |req|,        |
   |                            | atom_id_2 |req|                        |
   +----------------------------+----------------------------------------+
   | entity_poly                |                                        |
   |                            | entity_id |req|, nstd_monomer, type    |
   +----------------------------+----------------------------------------+
   | entity_poly_seq            |                                        |
   |                            | entity_id |req|, num |req|,            |
   |                            | mon_id |req|, hetero                   |
   +----------------------------+----------------------------------------+
   | entity                     |                                        |
   |                            | id |req|, pdbx_description             |
   +----------------------------+----------------------------------------+
   | entity_src_gen             |                                        |
   |                            | entity_id |req|,                       |
   |                            | pdbx_gene_src_scientific_name |req|    |
   +----------------------------+----------------------------------------+
   | entity_src_nat             |                                        |
   |                            | entity_id |req|,                       |
   |                            | pdbx_organism_scientific |req|         |
   +----------------------------+----------------------------------------+
   | entry                      |                                        |
   |                            | id |req|                               |
   +----------------------------+----------------------------------------+
   | pdbx_database_PDB_obs_spr  |                                        |
   |                            | id |req|, pdb_id |req|,                |
   |                            | replace_pdb_id |req|                   |
   +----------------------------+----------------------------------------+
   | pdbx_struct_assembly       |                                        |
   |                            | id |req|, details |req|                |
   +----------------------------+----------------------------------------+
   | pdbx_struct_assembly_gen   |                                        |
   |                            | assembly_id |req|,                     |
   |                            | oper_expression |req|,                 |
   |                            | asym_id_list |req|                     |
   +----------------------------+----------------------------------------+
   | pdbx_struct_oper_list      |                                        |
   |                            | id |req|, matrix[1][1] |req|,          |
   |                            | matrix[1][2] |req|, matrix[1][3] |req|,|
   |                            | matrix[2][1] |req|, matrix[2][2] |req|,|
   |                            | matrix[2][3] |req|, matrix[3][1] |req|,|
   |                            | matrix[3][2] |req|, matrix[3][3] |req|,|
   |                            | vector[1] |req|, vector[2] |req|,      |
   |                            | vector[3] |req|                        |
   +----------------------------+----------------------------------------+
   | struct_conf                |                                        |
   |                            | id |req|, conf_type_id |req|,          |
   |                            | beg_label_asym_id |req|,               |
   |                            | beg_label_comp_id |req|,               |
   |                            | beg_label_seq_id |req|,                |
   |                            | end_label_asym_id |req|,               |
   |                            | end_label_comp_id |req|,               |
   |                            | end_label_seq_id |req|                 |
   +----------------------------+----------------------------------------+
   | struct_conn                |                                        |
   |                            | conn_type_id |req|,                    |
   |                            | ptnr1_label_asym_id |req|,             |
   |                            | pdbx_ptnr1_PDB_ins_code,               |
   |                            | ptnr1_label_seq_id |req|,              |
   |                            | ptnr1_auth_seq_id,                     |
   |                            | pdbx_ptnr1_label_alt_id,               |
   |                            | ptnr1_label_atom_id |req|,             |
   |                            | ptnr1_label_comp_id |req|,             |
   |                            | ptnr1_symmetry,                        |
   |                            | ptnr2_label_asym_id |req|,             |
   |                            | pdbx_ptnr2_PDB_ins_code,               |
   |                            | ptnr2_label_seq_id |req|,              |
   |                            | ptnr2_auth_seq_id,                     |
   |                            | pdbx_ptnr2_label_alt_id,               |
   |                            | ptnr2_label_atom_id |req|,             |
   |                            | ptnr2_label_comp_id |req|,             |
   |                            | ptnr2_symmetry, pdbx_dist_value        |
   +----------------------------+----------------------------------------+
   | struct_sheet_range         |                                        |
   |                            | sheet_id |req|, id |req|,              |
   |                            | beg_label_asym_id |req|,               |
   |                            | beg_label_comp_id |req|,               |
   |                            | beg_label_seq_id |req|,                |
   |                            | end_label_asym_id |req|,               |
   |                            | end_label_comp_id |req|,               |
   |                            | end_label_seq_id |req|                 |
   +----------------------------+----------------------------------------+

atom_site
  Contains atom coordinates.
  Typically the largest table in a mmCIF file.
  wwPDB mmCIF files use fixed width columns for the data.

atom_site_anisotrop
  Contains anisotropic displacement data for atoms.
  While the specification for the **atom_site** category
  has provisions to include the anisotropic displacement data,
  in practice it is not.
  Consequently, ChimeraX only looks in the **atom_site_anisotrop**
  table for the anisotropic displacement data.
  wwPDB mmCIF files use fixed width columns for the data.

audit_conform
  Contains metadata about the CIF file.
  Can specify the CIF dicitionary and version the data conforms to.
  Used to guess about the styling.

audit_syntax
  Added by ChimeraX to hold the explicit metadata about styling with
  **case_sensitive_flag** and **fixed_width** keywords.
  With luck, this will turn into an official **audit_syntax** category.

chem_comp
  Contains information about the chemical components in the structure.
  Used for embedded residue templates.

chem_comp_bond
  Contains connectivity of chemical components.
  Used for embedded residue templates.
  Currently only present in "updated" PDB files from the PDBe.

entity
  Contains details "about the molecular entities that are
  present in the crystallographic structure."
  Used to extract description of chains.

entity_poly
  Tell if entity has non-standard monomers in it and thus, potentially,
  non-polymeric linkage.

entity_poly_seq
  Contains the sequence of residues in a chain.
  Used to know which residues to connect and where there are structural gaps.

entity_src_gen
  Contains "details of the source from which the entity was obtained
  in cases where the source was genetically manipulated."
  Used to extract scientific name of entities.

entity_src_nat
  Contains "details of the source from which the entity was obtained
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
  but that information is now in the **struct_sheet_range**
  data.

struct_conn
  Contains non-standard connectivity.
  Standard amino and nucleic acid connectivity is given by chemical
  component templates.

struct_sheet_range
  Contains strand residue ranges and associated sheets.
