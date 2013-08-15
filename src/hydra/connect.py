cpath = '/Users/goddard/ucsf/chimera2/src/hydra/components.cif'
ipath = '/Users/goddard/ucsf/chimera2/src/hydra/cindex'

def create_molecule_bonds(m, cpath = cpath, ipath = ipath):
    cids = set(m.residue_names)
    bt = chemical_component_bonds(cids, cpath, ipath)
    bonds = []
    n = m.atom_count()
    res = None
    for a in range(n):
        cid = m.residue_names[a]
        rnum = m.residue_nums[a]
        ch = m.chain_ids[a]
        if (cid,rnum,ch) != res:
            if not res is None:
                bonds.extend(template_bonds(ipairs, ai))
            ai = {}
            aindex, ipairs = bt[cid]
            res = (cid, rnum, ch)

        aname = m.atom_names[a]
        if aname in aindex:
            ai[aindex[aname]] = a
#        else:
#            print('Atom %s from residue %s has no template bonds' % (aname, str(res)))
    if not res is None:
        bonds.extend(template_bonds(ipairs, ai))
    bonds.extend(backbone_bonds(m))
    if bonds:
        from numpy import array, int32
        m.bonds = array(bonds, int32)

def template_bonds(ipairs, ai):
    bonds = []
    for i1,i2 in ipairs:
        if i1 in ai and i2 in ai:
            bonds.append((ai[i1],ai[i2]))
    return bonds

def backbone_bonds(m):
    bonds = []
    ajoin = ((b'C', b'N'), (b"O3'", b'P'))
    anames = sum(ajoin, ())
    bbatoms = {}
    for a in range(m.atom_count()):
        aname = m.atom_names[a]
        if aname in anames:
            rnum = m.residue_nums[a]
            ch = m.chain_ids[a]
            bbatoms[(rnum, ch, aname)] = a
    for (rnum, ch, aname), a1 in bbatoms.items():
        for n1,n2 in ajoin:
            if aname == n1:
                a2 = bbatoms.get((rnum+1, ch, n2))
                if not a2 is None:
                    bonds.append((a1,a2))
    return bonds

ccb = {}
def chemical_component_bonds(cids, cpath, ipath):
    global ccb
    ncids = tuple(cid for cid in cids if not cid in ccb)
    if len(ncids) == 0:
        return ccb
#    print('Reading bonds from file for %d residue types %s' % (len(ncids), str(ncids)))
    from numpy import fromfile, int32
    cindex = fromfile(ipath,int32)
    f = open(cpath)
    for cid in set(ncids):
        aindex, ipairs = component_bonds(cid, f, cindex)
        if not aindex is None:
            ccb[cid] = (aindex, ipairs)
    f.close()
    return ccb

def component_bonds(cid, f, cindex):
    i = component_id_index(cid)
    fi = cindex[i]
    if fi == -1:
        return None, None
    f.seek(fi,0)

    apairs = []
    foundb = False
    scid = cid.decode('utf-8')
    while True:
        line = f.readline()
        if line.startswith('_chem_comp_bond.'):
            foundb = True
        elif foundb:
            if line.startswith(scid):
                fields = line.split()
                a1,a2 = fields[1].strip('"').encode('utf-8'), fields[2].strip('"').encode('utf-8')
                apairs.append((a1,a2))
            else:
                break
        elif line == '' or (line.startswith('data_') and not line.startswith('data_' + scid)):
            break
    atoms = set([a1 for a1,a2 in apairs] + [a2 for a1,a2 in apairs])
    aindex = dict((a,i) for i,a in enumerate(atoms))
    ipairs = tuple((aindex[a1], aindex[a2]) for a1,a2 in apairs)
#    print('read %d bonds for %s, %s' % (len(ipairs), cid, str(apairs)))
    return aindex, ipairs

# connect atoms with bonds
def make_chemical_components_file_index(cpath, ipath):
    f = open(cpath)
    from numpy import empty, int32
    cp = empty((maxc**3,), int32)
    cp[:] = -1
    while True:
        b = f.tell()
        line = f.readline()
        if line.startswith('data_'):
            cid = line.rstrip()[5:8]
            i = component_id_index(cid)
            if i is None:
                print ('Chemical component "%s" contains a character other than A-Z,0-9,space' % cid)
            else:
                cp[i] = b
        elif line == '':
            break
    f.close()
    if not ipath is None:
        g = open(ipath, 'wb')
        g.write(cp.tostring())
        g.close()
    return cp

# Component id can be only uppercase letters and digits
cchars = b' 01234567890ABCDEFGHIJKLMNOPQRSTUVWXYZ'
maxc = len(cchars)
def component_id_index(cid):
    n = len(cchars)
    k = 0
    for c in cid:
        i = cchars.find(c)
        if i == -1:
            return None
        k = k*n + i
    return k

#make_chemical_components_file_index('components.cif', 'cindex')
