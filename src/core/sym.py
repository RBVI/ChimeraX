# Clone or copy a model.
def sym(session, molecules, assembly = None, clear = False, surface_only = False):
    for m in molecules:
        assem = pdb_assemblies(m)
        if clear:
            from .geometry import Place
            m.position = Place()
            for s in m.surfaces():
                s.position = Place()
        elif assembly is None:
            ainfo = '\n'.join(' %s = %s (%s copies)'
                              % (a.id,a.description,
                                 ','.join(str(len(ops)) for cids, expr, ops in a.chain_ops))
                              for a in assem)
            anames = ainfo if assem else "no assemblies"
            session.logger.info('Assemblies for %s:\n%s' % (m.name, anames))
        else:
            amap = dict((a.id, a) for a in assem)
            if not assembly in amap:
                from .errors import UserError
                raise UserError('Assembly "%s" not found, have %s'
                                % (assembly, ', '.join(a.id for a in assem)))
            a = amap[assembly]
            if surface_only:
                a.show_surfaces(m, session)
            else:
                a.show(m, session)

def register_sym_command():
    from .structure import AtomicStructuresArg
    from . import cli
    _sym_desc = cli.CmdDesc(
        required = [('molecules', AtomicStructuresArg)],
        keyword = [('assembly', cli.StringArg),
                   ('clear', cli.NoArg),
                   ('surface_only', cli.NoArg)],
        synopsis = 'create model copies')
    cli.register('sym', _sym_desc, sym)

def pdb_assemblies(m):
    if not hasattr(m, 'filename') or not m.filename.endswith('.cif'):
        return []
    if hasattr(m, 'assemblies'):
        return m.assemblies
    m.assemblies = alist = mmcif_assemblies(m.filename)
    return alist

def mmcif_assemblies(mmcif_path):
    table_names = ('_pdbx_struct_assembly',
                   '_pdbx_struct_assembly_gen',
                   '_pdbx_struct_oper_list',
                   '_pdbx_poly_seq_scheme',
                   '_pdbx_nonpoly_scheme')
    from . import mmcif
    assem, assem_gen, oper, cremap1, cremap2 = mmcif.read_mmcif_tables(mmcif_path, table_names)
    if assem is None or assem_gen is None or oper is None:
        return []

    name = assem.mapping('id', 'details')
    ids = list(name.keys())
    ids.sort()

    cops = assem_gen.fields(('assembly_id', 'oper_expression', 'asym_id_list'))
    chain_ops = {}
    for id, op_expr, cids in cops:
        chain_ops.setdefault(id,[]).append((cids.split(','), op_expr))

    ops = {}
    mat = oper.fields(('id',
                       'matrix[1][1]', 'matrix[1][2]', 'matrix[1][3]', 'vector[1]',
                       'matrix[2][1]', 'matrix[2][2]', 'matrix[2][3]', 'vector[2]',
                       'matrix[3][1]', 'matrix[3][2]', 'matrix[3][3]', 'vector[3]'))
    from .geometry import Place
    for id, m11,m12,m13,m14,m21,m22,m23,m24,m31,m32,m33,m34 in mat:
        ops[id] = Place(matrix = ((m11,m12,m13,m14),(m21,m22,m23,m24),(m31,m32,m33,m34)))

    cmap = chain_id_changes(cremap1, cremap2)

    alist = [Assembly(id, name[id], chain_ops[id], ops, cmap) for id in ids]
    return alist

#
# Assemblies described using mmCIF chain ids but Chimera uses author chain ids.
# Map author chain id and residue number to mmCIF chain id.
# Only include entries if chain id is changed.
#
def chain_id_changes(poly_seq_scheme, nonpoly_scheme):
    cmap = {}
    if not poly_seq_scheme is None:
        # Note the pdb_seq_num (and not the auth_seq_num) in this table corresponds to
        # auth_seq_id in the atom_site table.  Example 3efz.
        pcnc = poly_seq_scheme.fields(('asym_id', 'pdb_seq_num', 'pdb_strand_id'))
        cmap = dict(((auth_cid, int(auth_resnum)), mmcif_cid)
                    for mmcif_cid, auth_resnum, auth_cid in pcnc
                    if mmcif_cid != auth_cid and auth_resnum != '?')
    if not nonpoly_scheme is None:
        ncnc = nonpoly_scheme.fields(('asym_id', 'pdb_seq_num', 'pdb_strand_id'))
        ncmap = dict(((auth_cid, int(auth_resnum)), mmcif_cid)
                     for mmcif_cid, auth_resnum, auth_cid in ncnc
                     if mmcif_cid != auth_cid and auth_resnum != '?')
        cmap.update(ncmap)
    return cmap

class Assembly:
    def __init__(self, id, description, chain_ops, operator_table, chain_map):
        self.id = id
        self.description = description

        cops = []
        for chain_ids, operator_expr in chain_ops:
            products = parse_operator_expression(operator_expr)
            ops = operator_products(products, operator_table)
            cops.append((chain_ids, operator_expr, ops))
        self.chain_ops = cops	# Triples of chain id list, operator expression, operator matrices

        self.operator_table = operator_table
        # Chain map maps Chimera chain id, res name to mmcif chain id used in chain_ids
        self.chain_map = chain_map

    def show(self, mol, session):
        mols = self._molecule_copies(mol, session)
        for (chain_ids, op_expr, ops), m in zip(self.chain_ops, mols):
            included_atoms, excluded_atoms = self._partition_atoms(m.atoms, chain_ids)
            if len(excluded_atoms) > 0:
                # Hide chains that are not part of assembly
                excluded_atoms.displays = False
                excluded_atoms.unique_residues.ribbon_displays = False
            if not included_atoms.displays.all():
                # Show chains that have not atoms or ribbons shown.
                for mc, cid, catoms in included_atoms.by_chain:
                    if not catoms.displays.any() and not catoms.residues.ribbon_displays.any():
                        catoms.displays = True
            m.positions = ops

    def show_surfaces(self, mol, session):
        included_atoms, excluded_atoms = self._partition_atoms(mol.atoms, self._chain_ids())
        from .molsurf import surface_command
        surfs = surface_command(session, included_atoms)
        if len(excluded_atoms) > 0:
            surface_command(session, excluded_atoms, hide = True)
        for s in surfs:
            cid = s.atoms[0].residue.chain_id
            s.positions = self._chain_operators(cid)

    def _partition_atoms(self, atoms, chain_ids):
        mmcif_cids = mmcif_chain_ids(atoms, self.chain_map)
        from numpy import in1d, logical_not
        mask = in1d(mmcif_cids, chain_ids)
        included_atoms = atoms.filter(mask)
        logical_not(mask,mask)
        excluded_atoms = atoms.filter(mask)
        return included_atoms, excluded_atoms

    def _chain_ids(self):
        return sum((chain_ids for chain_ids, op_expr, ops in self.chain_ops), [])

    def _chain_operators(self, chain_id):
        cops = []
        for chain_ids, operator_expr, ops in self.chain_ops:
            if chain_id in chain_ids:
                cops.extend(ops)
        return cops

    def _molecule_copies(self, mol, session):
        copies = getattr(mol, '_sym_copies', [])
        nm = 1 + len(copies)
        n = len(self.chain_ops)
        if nm < n:
            # Create new copies
            mnew = [mol.copy('%s %d' % (mol.name,i)) for i in range(nm,n)]
            session.models.add(mnew)
            copies.extend(mnew)
            mol._sym_copies = copies
        elif nm > n:
            # Close extra copies
            session.models.close(copies[nm-n-1:])
            copies = copies[:nm-n-1]
            mol._sym_copies = copies
        mols = [mol] + copies
        return mols
            

def mmcif_chain_ids(atoms, chain_map):
    if len(chain_map) == 0:
        cids = atoms.residues.chain_ids
    else:
        r = atoms.residues
        from numpy import array
        cids = array([chain_map.get((cid,n), cid) for cid,n in zip(r.chain_ids, r.numbers)])
    return cids

def operator_products(products, oper_table):
    from .geometry import Places
    p = Places(tuple(oper_table[e] for e in products[0]))
    if len(products) > 1:
        p = p * operator_products(products[1:], oper_table)
    return p

# Example from 1m4x.cif (1,10,23)(61,62,69-88)
def parse_operator_expression(expr):
    product = []
    import re
    factors = [e for e in re.split('[()]', expr) if e]
    for f in factors:
        terms = f.split(',')
        elem = []
        for t in terms:
            dash = t.split('-')
            if len(dash) == 2:
                elem.extend(str(e) for e in range(int(dash[0]), int(dash[1])+1))
            else:
                elem.append(t)
        product.append(elem)
    return product
