def minimize_crosslinks(session, pbspec = None, iterations = 5, radius = None, color = None):
    pbg = parse_pseudobond_group_arg(pbspec, session)
    pbonds = pbg.pseudobonds
    mol_links = links_by_molecule(pbonds)
    mols = list(mol_links.keys())
    mols.sort(key = lambda m: m.name)
    from numpy import array, float64
    from . import align 
    for i in range(iterations):
        for m in mols:
            atom_pairs = mol_links[m]
            moving = array([a1.scene_coord for a1,a2 in atom_pairs], float64)
            fixed = array([a2.scene_coord for a1,a2 in atom_pairs], float64)
            tf, rms = align.align_points(moving, fixed)
            m.position = tf * m.position

    lengths = [pb.length for pb in pbonds]
    lengths.sort(reverse = True)
    print('%d crosslinks, lengths: %s' % (len(pbonds), ', '.join('%.1f' % d for d in lengths)))

    if radius:
        for pb in pbonds:
            pb.radius = radius
    if color:
        rgba = color.uint8x4()
        for pb in pbonds:
            pb.color = rgba
    pbg.update_graphics()	# TODO: pseudobonds should update automatically when atoms move.

def parse_pseudobond_group_arg(pbspec, session):
    from .pbgroup import PseudoBondGroup
    from .cli import UserError
    if pbspec is None:
        pbg_list = [m for m in session.models.list() if isinstance(m, PseudoBondGroup)]
        if len(pbg_list) == 0:
            raise UserError('No pseudobond group opened.')
        elif len(pbg_list) > 1:
            raise UserError('Multiple (%d) pseudobond groups opened, must specify one.' % len(pbg_list))
        pbg = pbg_list[0]
    else:
        pbg_list = pbspec.evaluate(session).models
        if len(pbg_list) == 0:
            raise UserError('No pseudobond group specified by "%s".' % str(pbspec))
        elif len(pbg_list) > 1:
            raise UserError('Multiple (%d) pseudobond groups specifed, must specify one.' % len(pbg_list))
        pbg = pbg_list[0]
    return pbg

def links_by_molecule(pblist):
    mol_links = {}
    for pb in pblist:
        a1, a2 = pb.atoms
        m1, m2 = a1.molecule, a2.molecule
        if m1 != m2:
            mol_links.setdefault(m1,[]).append((a1,a2))
            mol_links.setdefault(m2,[]).append((a2,a1))
    return mol_links

def register_crosslink_command():
    from . import cli
    from .atomspec import AtomSpecArg
    from .color import ColorArg
    desc = cli.CmdDesc(optional = [('pbspec', AtomSpecArg)],
                       keyword = [('iterations', cli.IntArg),
                                  ('radius', cli.FloatArg),
                                  ('color', ColorArg),
                              ])
    cli.register('crosslink', desc, minimize_crosslinks)
