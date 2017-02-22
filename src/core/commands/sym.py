# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2016 Regents of the University of California.
# All rights reserved.  This software provided pursuant to a
# license agreement containing restrictions on its disclosure,
# duplication and use.  For details see:
# http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html
# This notice must be embedded in or attached to all copies,
# including partial copies, of the software or any revisions
# or derivations thereof.
# === UCSF ChimeraX Copyright ===

def sym(session, structures,
        symmetry = None, center = None, axis = None, coordinate_system = None, assembly = None,
        copies = False, new_model = None, surface_only = False, resolution = None, grid_spacing = None):
    '''
    Show molecular assemblies of molecular models defined in mmCIF files.
    These can be subassemblies or symmetrical copies with individual chains 
    placed according to matrices specified in the mmCIF file.

    Parameters
    ----------
    structures : list of AtomicStructure
      List of structures to show as assemblies.
    symmetry : cli.Symmetry
      Desired symmetry to display
    center : cli.Center
      Center of symmetry.  Default 0,0,0.
    axis : Axis
      Axis of symmetry.  Default z.
    coordinate_system : Place
      Transform mapping coordinates for center and axis arguments to scene coordinates.
    assembly : string
      The name of assembly in the mmCIF file. If this parameter is None
      then the names of available assemblies are printed in log.
    copies : bool
      Whether to make copies of the molecule chains.  If copies are not made
      then graphical instances of the original molecule are used.  Copies are needed
      to give different colors or styles to each copy.  When copies are made a new model
      with submodels for each copy are created.
    new_model : bool
      Copy the structure to create the assembly.  Default is True if an assembly is specified
      and false if the symmetry option is given.  If the copies option is true then
      new_model is automatically true.  If copies is false and new_model is true then
      one copy of the structure for each set of position matrices is used, and additional
      copies use graphical instances.  If copies is false and new_model is false then
      instances of the original structure is used.  If there is more than one set of
      position matrices, then copies are required and an error message is given.
    surface_only : bool
      Instead of showing instances of the molecule, show instances
      of surfaces of each chain.  The chain surfaces are computed if
      they do not already exist.  It is an error to request surfaces only
      and to request copies.
    resolution : float
      Resolution for computing surfaces when surface_only is true.
    grid_spacing : float
      Grid spacing for computing surfaces when surface_only is true.
    '''
    if len(structures) == 0:
        from ..errors import UserError
        raise UserError('No structures specified.')

    if surface_only and copies:
        from ..errors import UserError
        raise UserError('Cannot use both copies and surfacesOnly.')

    if new_model is None:
        if assembly is not None:
            new_model = True
        elif symmetry is not None:
            new_model = False
            
    if symmetry is not None:
        if assembly is not None:
            from ..errors import UserError
            raise UserError('Cannot specify explicit symmetry and the assembly option.')
        transforms = symmetry.positions(center, axis, coordinate_system, structures[0])
        show_symmetry(structures, symmetry.group, transforms, copies, new_model, surface_only,
                      resolution, grid_spacing, session)
        return
            
    for m in structures:
        assem = pdb_assemblies(m)
        if assembly is None:
            ainfo = '\n'.join(' %s = %s (%s)' % (a.id,a.description,a.copy_description(m))
                              for a in assem)
            anames = ainfo if assem else "no assemblies"
            session.logger.info('Assemblies for %s:\n%s' % (m.name, anames))
            for a in assem:
                a.create_selector(m)
        else:
            amap = dict((a.id, a) for a in assem)
            if not assembly in amap:
                from ..errors import UserError
                raise UserError('Assembly "%s" not found, have %s'
                                % (assembly, ', '.join(a.id for a in assem)))
            a = amap[assembly]
            if copies:
                a.show_copies(m, surface_only, resolution, grid_spacing, session)
            elif surface_only:
                a.show_surfaces(m, resolution, grid_spacing, new_model, session)
            else:
                a.show(m, new_model, session)
            if new_model:
                m.display = False

def sym_clear(session, structures = None):
    '''
    Remove copies of structures that were made with sym command.

    Parameters
    ----------
    structures : list of AtomicStructure
      List of structures to for which to remove copies.
    '''
    if structures is None:
        from ..atomic import all_structures
        structures = all_structures(session)
    for m in structures:
        from ..geometry import Places
        m.positions = Places([m.position])	# Keep only first position.
        for s in m.surfaces():
            s.positions = Places([s.position])

def register_command(session):
    from . import CmdDesc, register, AtomicStructuresArg, StringArg, FloatArg
    from . import CenterArg, AxisArg, SymmetryArg, CoordSysArg, BoolArg
    desc = CmdDesc(
        required = [('structures', AtomicStructuresArg)],
        optional = [('symmetry', SymmetryArg)],
        keyword = [('center', CenterArg),
                   ('axis', AxisArg),
                   ('coordinate_system', CoordSysArg),
                   ('assembly', StringArg),
                   ('copies', BoolArg),
                   ('new_model', BoolArg),
                   ('surface_only', BoolArg),
                   ('resolution', FloatArg),
                   ('grid_spacing', FloatArg)],
        synopsis = 'create model copies')
    register('sym', desc, sym, logger=session.logger)
    desc = CmdDesc(
        optional = [('structures', AtomicStructuresArg)],
        synopsis = 'Remove model copies')
    register('sym clear', desc, sym_clear, logger=session.logger)

def show_symmetry(structures, sym_name, transforms, copies, new_model, surface_only,
                  resolution, grid_spacing, session):
    name = '%s %s' % (','.join(s.name for s in structures), sym_name)
    if copies and not new_model:
        from ..models import Model
        g = Model(name, session)
        for i, tf in enumerate(transforms):
            if len(structures) > 1:
                # Add grouping model if more the one model is being copied
                ci = Model('copy %d' % (i+1), session)
                ci.position = tf
                g.add([ci])
                copies = [m.copy() for m in structures]
                for c,m in zip(copies, structures):
                    c.position = m.scene_position
                ci.add(copies)
            else:
                m0 = structures[0]
                c = m0.copy()
                c.position = tf * m0.scene_position
                g.add([c])
        session.models.add([g])
    else:
        if new_model:
            from ..models import Model
            group = Model(name, session)
            mols = [m.copy() for m in structures]
            group.add(mols)
            session.models.add([group])
        else:
            mols = structures
        # Instancing
        for m in mols:
            # Transforms are in scene coordinates, so convert to molecule coordinates
            spos = m.scene_position
            symops = transforms if spos.is_identity() else transforms.transform_coordinates(spos)
            if surface_only:
                from .surface import surface
                surfs = surface(session, m.atoms, grid_spacing = grid_spacing, resolution = resolution)
                for s in surfs:
                    s.positions =  s.positions * symops
            else:
                m.positions = m.positions * symops

    if copies or new_model:
        for s in structures:
            s.display = False
            
def pdb_assemblies(m):
    if not hasattr(m, 'filename') or not m.filename.endswith('.cif'):
        return []
    if hasattr(m, 'assemblies'):
        return m.assemblies
    m.assemblies = alist = mmcif_assemblies(m)
    return alist

def mmcif_assemblies(model):
    table_names = ('pdbx_struct_assembly',
                   'pdbx_struct_assembly_gen',
                   'pdbx_struct_oper_list',
                   'pdbx_poly_seq_scheme',
                   'pdbx_nonpoly_scheme')
    from ..atomic import mmcif
    assem, assem_gen, oper, cremap1, cremap2 = mmcif.get_mmcif_tables(model.filename, table_names)
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
    from ..geometry import Place
    for id, m11,m12,m13,m14,m21,m22,m23,m24,m31,m32,m33,m34 in mat:
        ops[id] = Place(matrix = ((m11,m12,m13,m14),(m21,m22,m23,m24),(m31,m32,m33,m34)))

    cmap = chain_id_changes(cremap1, cremap2)

    alist = [Assembly(id, name[id], chain_ops[id], ops, cmap) for id in ids]
    return alist

#
# Assemblies described using mmCIF chain ids but ChimeraX uses author chain ids.
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
        # Chain map maps ChimeraX (chain id, res number) to mmcif chain id used in chain_ids
        self.chain_map = chain_map

    def show(self, mol, new_model, session):
        mols = self._molecule_copies(mol, new_model, session)
        for (chain_ids, op_expr, ops), m in zip(self.chain_ops, mols):
            included_atoms, excluded_atoms = self._partition_atoms(m.atoms, chain_ids)
            if len(excluded_atoms) > 0:
                if new_model:
                    excluded_atoms.delete()
                else:
                    # Hide chains that are not part of assembly
                    excluded_atoms.displays = False
                    excluded_atoms.unique_residues.ribbon_displays = False
            self._show_atoms(included_atoms)
            m.positions = ops

    def _show_atoms(self, atoms):
        if not atoms.displays.all():
            # Show chains that have not atoms or ribbons shown.
            for mc, cid, catoms in atoms.by_chain:
                if not catoms.displays.any() and not catoms.residues.ribbon_displays.any():
                    catoms.displays = True

    def show_surfaces(self, mol, res, grid_spacing, new_model, session):
        if new_model:
            m = mol.copy('%s assembly %s' % (mol.name, self.id))
            m.ignore_assemblies = True
            session.models.add([m])
        else:
            m = mol
        included_atoms, excluded_atoms = self._partition_atoms(m.atoms, self._chain_ids())
        if new_model:
            excluded_atoms.delete()
        from .surface import surface, surface_hide
        surfs = surface(session, included_atoms, grid_spacing = grid_spacing, resolution = res)
        if not new_model and len(excluded_atoms) > 0:
            from ..objects import Objects
            surface_hide(session, Objects(atoms = excluded_atoms))
        for s in surfs:
            mmcif_cid = mmcif_chain_ids(s.atoms[:1], self.chain_map)[0]
            s.positions = self._chain_operators(mmcif_cid)

    def show_copies(self, mol, surface_only, resolution, grid_spacing, session):
        mlist = []
        for chain_ids, op_expr, ops in self.chain_ops:
            for pos in ops:
                m = mol.copy()
                m.ignore_assemblies = True
                mlist.append(m)
                m.position = pos
                included_atoms, excluded_atoms = self._partition_atoms(m.atoms, chain_ids)
                if len(excluded_atoms) > 0:
                    excluded_atoms.delete()
                self._show_atoms(included_atoms)

        g = session.models.add_group(mlist)[0]
        g.name = '%s assembly %s' % (mol.name, self.id)

        if surface_only:
            from .surface import surface
            for m in mlist:
                surface(session, m.atoms, grid_spacing = grid_spacing, resolution = resolution)

        mol.display = False

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
        from ..geometry import Places
        return Places(cops)

    def _molecule_copies(self, mol, new_model, session):
        n = len(self.chain_ops)
        if not new_model:
            if n > 1:
                from ..errors import UserError
                raise UserError('Assembly requires new model because'
                                'it uses more than one set of positioning matrices.')
            else:
                return [mol]
        # Create copies
        name = '%s assembly %s' % (mol.name, self.id)
        if n > 1:
            from ..models import Model
            group = Model(name, session)
            mcopies = [mol.copy('%s %d' % (mol.name,i+1)) for i in range(n)]
            group.add(mcopies)
            addm = [group]
        else:
            mcopies = addm = [mol.copy(name)]
        for m in mcopies:
            m.ignore_assemblies = True
        session.models.add(addm)
        return mcopies

    def copy_description(self, mol):
        atoms = mol.atoms
        groups = []
        for cids, expr, ops in self.chain_ops:
            # Convert mmcif chain ids to author chain ids
            incl_atoms, excl_atoms = self._partition_atoms(atoms, cids)
            author_cids = ','.join(incl_atoms.unique_chain_ids)
            copy = 'copy' if len(ops) == 1 else 'copies'
            groups.append('%d %s of chains %s' % (len(ops), copy, author_cids))
        return ', '.join(groups)

    def create_selector(self, mol):
        if self._is_subassembly():
            name = self.id
            sel_name = ('A' + name) if is_integer(name) else name
            def _selector(session, models, results, self=self, mol=mol):
                atoms = self._subassembly_atoms(mol)
                results.add_atoms(atoms)
            from . import register_selector
            register_selector(sel_name, _selector)

    def _is_subassembly(self):
        cops = self.chain_ops
        if len(cops) != 1:
            return False
        chain_ids, op_expr, ops = cops[0]
        if len(ops) != 1:
            return False
        return True
        
    def _subassembly_atoms(self, mol):
        chain_ids = self.chain_ops[0][0]
        included_atoms, excluded_atoms = self._partition_atoms(mol.atoms, chain_ids)
        return included_atoms

def is_integer(s):
    try:
        int(s)
    except:
        return False
    return True

def mmcif_chain_ids(atoms, chain_map):
    if len(chain_map) == 0:
        cids = atoms.residues.chain_ids
    else:
        r = atoms.residues
        from numpy import array
        cids = array([chain_map.get((cid,n), cid) for cid,n in zip(r.chain_ids, r.numbers)])
    return cids

def operator_products(products, oper_table):
    from ..geometry import Places
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
