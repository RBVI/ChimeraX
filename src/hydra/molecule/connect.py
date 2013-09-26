def molecule_bonds(molecule):
    '''
    Return bonds derived from residue templates where each bond is a pair of atom numbers.
    Returned bonds are an N by 2 numpy array.
    '''
    global bond_templates
    if bond_templates is None:
        bond_templates = Bond_Templates()
    from time import time
    t0 = time()
    bonds, missing = bond_templates.molecule_bonds(molecule)
    t1 = time()
    print('Computed', len(bonds), 'bonds for', molecule.name, 'in', '%.3f' % (t1-t0), 'seconds')
    return bonds, missing

bond_templates = None

class Bond_Templates:
    '''
    Use reference data describing standard PDB chemical components
    to determine which atoms are bonded in a molecule.
    '''

    def __init__(self, templates_file = None):

        if templates_file is None:
            from os.path import join, dirname
            templates_file = join(dirname(__file__), 'bond_templates')
        self.templates_file = templates_file

        self.cc_index = None       # Index into all bonds list for each chemical component
        self.all_bonds = None      # Bonds for all chemical components.
                                   #   Array of atom names, a pair for each bond, empty name separates chemical components.
        self.cc_bond_table = {}    # Bond table for each chemical component

    def molecule_bonds(self, molecule):

        from .. import _image3d
        if self.cc_index is None:
            self.read_templates_file()
            _image3d.initialize_bond_templates(self.cc_index, self.all_bonds, cc_chars.decode('utf-8'))
            print ('initialized bond table')
        m = molecule
        return _image3d.molecule_bonds(m.residue_names, m.residue_nums, m.chain_ids, m.atom_names)

    def molecule_bonds_orig(self, molecule):

        m = molecule
        unique_rnames = set(m.residue_names)
        bt = self.chemical_component_bond_tables(unique_rnames)

        bonds = []
        res = index_pairs = None
        atom_num = {}
        missing_template = set()
        anames, rnames, rnums, cids = m.atom_names, m.residue_names, m.residue_nums, m.chain_ids
        for a in range(m.atom_count()):
            rname = rnames[a]
            rnum = rnums[a]
            cid = cids[a]
            if (rname,rnum,cid) != res:
                if not index_pairs is None:
                    bonds.extend(self.template_bonds(index_pairs, atom_num))
                atom_num.clear()
                if rname in bt:
                    aindex, index_pairs = bt[rname]
                else:
                    aindex = index_pairs = None
                    missing_template.add(rname)
                res = (rname, rnum, cid)

            aname = anames[a]
            if aindex and aname in aindex:
                atom_num[aindex[aname]] = a
#            else:
#                print('Atom %s from residue %s has no template bonds' % (aname, str(res)))

        if not index_pairs is None:
            bonds.extend(self.template_bonds(index_pairs, atom_num))

        bonds.extend(self.backbone_bonds(m))

        from numpy import array, int32, empty
        ba = array(bonds, int32) if bonds else empty((0,2), int32)

        return ba, missing_template

    def template_bonds(self, index_pairs, atom_num):
        bonds = []
        for i1,i2 in index_pairs:
            if i1 in atom_num and i2 in atom_num:
                bonds.append((atom_num[i1], atom_num[i2]))
        return bonds

    def backbone_bonds(self, m):
        '''Connect consecutive residues in proteins and nucleic acids.'''
        bonds = []
        ajoin = ((b'C', b'N'), (b"O3'", b'P'))
        anames = sum(ajoin, ())
        bbatoms = {}
        for a in range(m.atom_count()):
            aname = m.atom_names[a]
            if aname in anames:
                rnum = m.residue_nums[a]
                cid = m.chain_ids[a]
                bbatoms[(rnum, cid, aname)] = a
        for (rnum, cid, aname), a1 in bbatoms.items():
            for n1,n2 in ajoin:
                if aname == n1:
                    a2 = bbatoms.get((rnum+1, cid, n2))
                    if not a2 is None:
                        bonds.append((a1,a2))
        return bonds

    def chemical_component_bond_tables(self, rnames):
        '''Create template bond tables for specified chemical components'''
        ccbt = self.cc_bond_table
        new_rnames = set(rname for rname in rnames if not rname in ccbt)
        if len(new_rnames) == 0:
            return ccbt

#        print('Reading bonds from file for %d residue types %s' % (len(new_rnames), str(new_rnames)))
        if self.cc_index is None:
            self.read_templates_file()
        cci,blist = self.cc_index, self.all_bonds

        for rname in new_rnames:
            i = component_index(rname)
            if i is None:
                continue
            apairs = []
            bi = cci[i]
            if bi != -1:
                while blist[bi]:
                    apairs.append((blist[bi], blist[bi+1]))
                    bi += 2
#            print('Read %s bonds %s' % (str(rname), str(apairs)))
            atoms = set([a1 for a1,a2 in apairs] + [a2 for a1,a2 in apairs])
            aindex = dict((a,i) for i,a in enumerate(atoms))
            ipairs = tuple((aindex[a1], aindex[a2]) for a1,a2 in apairs)
            ccbt[rname] = (aindex, ipairs)

        return ccbt

    def read_templates_file(self):

        bt = open(self.templates_file, 'rb')
        from numpy import fromstring, int32
        self.cc_index = fromstring(bt.read(4*n_cc_chars**3), int32)
        self.all_bonds = fromstring(bt.read(), 'S4')
        bt.close()

def write_template_bonds_file(components_cif_path, template_bonds_path):
    '''
    For each compound in the PDB chemical components file (components.cif)
    record the bonds as pairs of atom names.
    '''
    f = open(components_cif_path)
    from numpy import empty, int32, array
    cp = empty((n_cc_chars**3,), int32)
    cp[:] = -1
    bonds = []
    while True:
        rname, rbonds = next_mmcif_bonds(f)
        if rname is None:
            break
        i = component_index(rname)
        cp[i] = len(bonds)
        bonds.extend(sum(rbonds,()) + (b'',))
#        print('%s %d %d' % (str(rname), i, len(rbonds)))
    f.close()
    ba = array(bonds, 'S4')
    bt = open(template_bonds_path, 'wb')
    bt.write(cp.tostring())
    bt.write(ba.tostring())
    bt.close()
    return cp, ba

def next_mmcif_bonds(f):
    apairs = []
    cid = None
    foundb = False
    while True:
        line = f.readline()
        if line.startswith('_chem_comp_bond.'):
            foundb = True
        elif foundb:
            if line.startswith('#'):
                if len(apairs) > 0:
                    break
                else:
                    foundb = False
                    continue
            fields = line.split()
            cid = fields[0].encode('utf-8')
            a1,a2 = fields[1].strip('"').encode('utf-8'), fields[2].strip('"').encode('utf-8')
            apairs.append((a1,a2))
        elif line == '':
            break
    return cid, apairs

# Component id can be only uppercase letters and digits
cc_chars = b' 01234567890ABCDEFGHIJKLMNOPQRSTUVWXYZ'
n_cc_chars = len(cc_chars)
def component_index(rname):
    '''Map every 3 character chemical component name to an integer.'''
    n = len(cc_chars)
    k = 0
    for c in rname:
        i = cc_chars.find(c)
        if i == -1:
            return None
        k = k*n + i
    return k

if __name__ == '__main__':
    write_mmcif_bonds_table('components.cif', 'bond_templates')
