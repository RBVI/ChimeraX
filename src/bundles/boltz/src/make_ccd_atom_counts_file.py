def make_ccd_atom_counts_file():
    '''This requires rdkit to unpickle the Boltz ccd.pkl file.'''
    from os.path import expanduser, exists
    counts_path = expanduser('~/.boltz/ccd_atom_counts.npz')
    if exists(counts_path):
        return
    ccd_path = expanduser('~/.boltz/ccd.pkl')
    import pickle
    with open(ccd_path, 'rb') as f:
        ccd_mols = pickle.load(f)
    from rdkit.Chem.rdMolDescriptors import CalcNumHeavyAtoms
    ccd_counts = {ccd:CalcNumHeavyAtoms(mol) for ccd, mol in ccd_mols.items()}
    from numpy import array, savez, int32
    ccds = array(tuple(ccd_counts.keys()))
    counts = array(tuple(ccd_counts.values()), int32)
    savez(counts_path, ccds = ccds, counts = counts)

make_ccd_atom_counts_file()
