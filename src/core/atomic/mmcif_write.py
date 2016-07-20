# vim: set expandtab shiftwidth=4 softtabstop=4:
"""
Write mmCIF files.

Currently only writes a minimal atom_site table.
No sequence, no secondary structure, no assemblies, no alt locs.
"""

def write_mmcif(session, path, models, **kw):
    from . import Structure
    if models is None:
        models = session.models.list(type = Structure)
        if len(models) != 1:
            from ..errors import UserError
            raise UserError('No model specified for saving map')

    if len(models) != 1:
        from ..errors import UserError
        raise UserError('Can only write 1 model to an mmCIF file, got %d' % len(models))

    for m in models:
        if not isinstance(m, Structure):
            from ..errors import UserError
            raise UserError('Model %s is not an atomic structure' % m.name)

    atoms = models[0].atoms
    
#    from . import concatenate, Atoms
#    atoms = concatenate([m.atoms for m in models], Atoms)
    
    xyz = atoms.coords
    n = len(xyz)
    elem = atoms.element_names
    aname = atoms.names
    occ = atoms.occupancy
    bfact = atoms.bfactors
    res = atoms.residues
    rname = res.names
    cid = res.chain_ids  # string fields need "." if blank.
    rnum = res.numbers
    eid = entity_ids(atoms, res, rname)

    atom_site_header = '''loop_
_atom_site.id 
_atom_site.type_symbol 
_atom_site.label_atom_id 
_atom_site.label_comp_id 
_atom_site.label_asym_id 
_atom_site.label_entity_id 
_atom_site.label_seq_id 
_atom_site.Cartn_x 
_atom_site.Cartn_y 
_atom_site.Cartn_z 
_atom_site.occupancy 
_atom_site.B_iso_or_equiv 
'''

    lines = [('%s %s %s %s %s %d %d %.3f %.3f %.3f %.2f %.2f' %
              (a+1, elem[a], aname[a], rname[a], cid[a], eid[a], rnum[a],
               xyz[a,0], xyz[a,1], xyz[a,2], occ[a], bfact[a]))
             for a in range(n)]
    lines.append('#')

    text = atom_site_header + '\n'.join(lines) + '\n'
    
    f = open(path, 'w')
    f.write(text)
    f.close()

# To determine entity id, use residue name if in_chain is false, use sequence if in_chain is true.
def entity_ids(atoms, residues, rname):
    n = len(residues)
    from numpy import empty, int32
    eids = empty((n,), int32)

    inch = atoms.in_chains
    ch = residues.filter(inch).chains  # chain.characters is sequence.

    entities = {}	# Map sequence or residue name to entity id
    ci = 0		# Index of atoms in chains
    seqs = {}		# Map chain to sequence to avoid creating sequence string for each atom
    for i in range(n):
        if inch[i]:
            chain = ch[ci]
            ci += 1
            seq = seqs.get(chain)
            if seq is None:
                seqs[chain] = seq = chain.characters
            eid = entities.get(seq)
            if eid is None:
                entities[seq] = eid = len(entities) + 1
        else:
            r = ('residue', rname[i])
            eid = entities.get(r)
            if eid is None:
                entities[r] = eid = len(entities) + 1
        eids[i] = eid

    return eids
