..  vim: set expandtab shiftwidth=4 softtabstop=4:

.. default-domain:: cpp

Generate structure connectivity
===============================

The primary public function offered by the connectivity
shared library is :func:`connect_structure` (declared in
connectivity/connect.h along with other supporting
functions/classes):

.. function:: void connect_structure(AtomicStructure *as, std::vector<Residue *> *chain_starters, std::vector<Residue *> *chain_enders, std::set<Atom *> *conect_atoms, std::set<MolResId> *mod_res)

    :param as: AtomicStructure to create Bonds for
    :param chain_starters: Residues that start polymer chains
    :param chain_enders: Residues that end polymer chains
    :param conect_atoms: Atoms whose connectivity has been specified
        a priori (e.g. in PDB CONECT records).  Bonds will not be
        generated for such atoms and are the responsibility of the caller
    :param mod_res: MolResIds for residues with standard names that
        nonetheless have non-standard connectivity (e.g. those found in
        PDB MODRES records) and that therefore should not have template
        connectivity applied.  A MolResId can be constructed from a 
        Residue pointer.  The :func:`standard_residue` function can be
        used to determine if a residue name is considered standard
        (arg is a string).

