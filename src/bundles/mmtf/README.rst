:::::::::::::
ChimeraX-MMTF
:::::::::::::

The ChimeraX-MMTF bundle implements support for the `RCSB <http://www.rcsb.org/>`_'s `MMTF <http://mmtf.rcsb.org>`_ format.

Implementation Notes
--------------------

1. MMTF has intra-residue metal coordination bonds listed as covalent bonds
   (*e.g.*, HEM residues),
   but is missing inter-residue metal coordinate bonds.
   * Need to post-process and find metal coordination bonds from Python to fix
     intra-residue bonds
     * find_and_add_metal_coordination_bonds(mol)
   * need to push for all metal coordination bonds to be provided

2. Opening 3j3q is *much* slower with MMTF than reading the mmCIF file because
   the MMTF Python library creates Python objects for everything.
   * The fix is to use a C library version of MMTF when it is available
     and never create the Python objects

3. Haven't implemented sequence support yet because MMTF only gives the
   single letter codes for each residue and ChimeraX wants the three letter
   codes.
   * Eventually will need:
     *  mol.set_input_seq_info(auth_chain_id, seqres)
     *  mol.input_seq_source = "MMTF"

4. Need to support gaps
