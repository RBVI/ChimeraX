def download_weights_and_ccd():
    '''Call Boltz code to fetch the model weights and CCD database.'''
    from os.path import expanduser, exists
    data_path = expanduser('~/.boltz')
    if not exists(data_path):
        from os import mkdir
        mkdir(data_path)

    from boltz.main import download_boltz2
    from pathlib import Path
    download_boltz2(Path(data_path))

def make_ccd_atom_counts_file():
    '''This requires rdkit to unpickle the Boltz ccd.pkl file.'''
    from os.path import expanduser, exists, join
    counts_path = expanduser('~/.boltz/ccd_atom_counts_boltz2.npz')
    if exists(counts_path):
        return
    ccd_dir = expanduser('~/.boltz/mols')
    from os import listdir
    ccd_filenames = listdir(ccd_dir)
    print(f'Making CCD atom counts table for {len(ccd_filenames)} in {ccd_dir}')
    ccd_counts = {}
    for ccd_filename in ccd_filenames:
        if ccd_filename.endswith('.pkl'):
            import pickle
            with open(join(ccd_dir,ccd_filename), 'rb') as f:
                mol = pickle.load(f)
            from rdkit.Chem.rdMolDescriptors import CalcNumHeavyAtoms
            ccd = ccd_filename.replace('.pkl', '')
            ccd_counts[ccd] = CalcNumHeavyAtoms(mol)
    from numpy import array, savez, int32
    ccds = array(tuple(ccd_counts.keys()))
    counts = array(tuple(ccd_counts.values()), int32)
    savez(counts_path, ccds = ccds, counts = counts)

download_weights_and_ccd()
make_ccd_atom_counts_file()
