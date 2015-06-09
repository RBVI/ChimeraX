def crosslink(session, pbgroups = None, color = None, radius = None, minimize = None, iterations = 10):

    if pbgroups is None:
        from . import pbgroup
        pbgroups = pbgroup.all_pseudobond_groups(session.models)

    if len(pbgroups) == 0:
        from .cli import UserError        
        raise UserError('No pseudobond groups specified.')

    from .molecule import concatenate
    pbonds = concatenate([pbg.pseudobonds for pbg in pbgroups])

    if color:
        rgba = color.uint8x4()
        for pb in pbonds:
            pb.color = rgba

    if radius:
        for pb in pbonds:
            pb.radius = radius

    if minimize:
        mols = minimize
        if len(mols) == 0:
            from .cli import UserError        
            raise UserError('No structures specified for minimizing crosslinks.')
        mol_links, mol_pbonds = links_by_molecule(pbonds, mols)
        if len(mol_links) == 0:
            from .cli import UserError        
            raise UserError('No pseudobonds to minimize for specified molecules.')
        if len(mols) == 1:
            iterations = min(1,iterations)
        from numpy import array, float64
        from . import align
        for i in range(iterations):
            for m in mols:
                if m in mol_links:
                    atom_pairs = mol_links[m]
                    moving = array([a1.scene_coord for a1,a2 in atom_pairs], float64)
                    fixed = array([a2.scene_coord for a1,a2 in atom_pairs], float64)
                    tf, rms = align.align_points(moving, fixed)
                    m.position = tf * m.position

        lengths = [pb.length for pb in mol_pbonds]
        lengths.sort(reverse = True)
        lentext = ', '.join('%.1f' % d for d in lengths)
        session.logger.info('%d crosslinks, lengths: %s' % (len(mol_pbonds), lentext))

    for pbg in pbgroups:
        pbg.update_graphics()	# TODO: pseudobonds should update automatically

def links_by_molecule(pbonds, mols):
    mol_links = {}
    mol_pbonds = set()
    mset = set(mols)
    for pb in pbonds:
        a1, a2 = pb.atoms
        m1, m2 = a1.molecule, a2.molecule
        if m1 != m2:
            if m1 in mset:
                mol_links.setdefault(m1,[]).append((a1,a2))
                mol_pbonds.add(pb)
            if m2 in mset:
                mol_links.setdefault(m2,[]).append((a2,a1))
                mol_pbonds.add(pb)
    return mol_links, mol_pbonds

def register_crosslink_command():
    from . import cli
    from .pbgroup import PseudoBondGroupsArg
    from .color import ColorArg
    from .structure import AtomicStructuresArg
    desc = cli.CmdDesc(optional = [('pbgroups', PseudoBondGroupsArg)],
                       keyword = [('color', ColorArg),
                                  ('radius', cli.FloatArg),
                                  ('minimize', AtomicStructuresArg),
                                  ('iterations', cli.IntArg),
                              ])
    cli.register('crosslink', desc, crosslink)
