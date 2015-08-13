def crosslink(session, pbgroups = None, color = None, radius = None, minimize = None, iterations = 10, frames = None):

    if pbgroups is None:
        from .. import pbgroup
        pbgroups = pbgroup.all_pseudobond_groups(session.models)

    if len(pbgroups) == 0:
        from ..errors import UserError        
        raise UserError('No pseudobond groups specified.')

    from ..molecule import concatenate
    pbonds = concatenate([pbg.pseudobonds for pbg in pbgroups])

    if color:
        rgba = color.uint8x4()
        for pb in pbonds:
            pb.color = rgba

    if radius:
        for pb in pbonds:
            pb.radius = radius

    if minimize:
        minimize_link_lengths(minimize, pbonds, iterations, frames, session)

def minimize_link_lengths(mols, pbonds, iterations, frames, session):
    if len(mols) == 0:
        from ..errors import UserError        
        raise UserError('No structures specified for minimizing crosslinks.')
    mol_links, mol_pbonds = links_by_molecule(pbonds, mols)
    if len(mol_links) == 0:
        from ..errors import UserError        
        raise UserError('No pseudobonds to minimize for specified molecules.')
    if len(mols) == 1:
        iterations = min(1,iterations)
    if not frames is None:
        pos0 = dict((m,m.position) for m in mols)
    from numpy import array, float64
    from .. import align
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

    if not frames is None:
        for m in mols:
            interpolate_position(m, pos0[m], m.position, frames, session.main_view)

def links_by_molecule(pbonds, mols):
    mol_links = {}
    mol_pbonds = set()
    mset = set(mols)
    for pb in pbonds:
        a1, a2 = pb.atoms
        m1, m2 = a1.structure, a2.structure
        if m1 != m2:
            if m1 in mset:
                mol_links.setdefault(m1,[]).append((a1,a2))
                mol_pbonds.add(pb)
            if m2 in mset:
                mol_links.setdefault(m2,[]).append((a2,a1))
                mol_pbonds.add(pb)
    return mol_links, mol_pbonds

class interpolate_position:

    def __init__(self, model, pos0, pos1, frames, view):
        self.model = model
        self.pos0 = pos0
        self.pos1 = pos1
        self.frames = frames
        self.frame = 1
        self.view = view

        b = model.bounds()
        if b is None:
            model.position = pos1
        else:
            center = 0.5*(b.xyz_min + b.xyz_max)
            self.c0, self.c1 = pos0*center, pos1*center
            self.axis, self.angle = (pos1*pos0.inverse()).rotation_axis_and_angle()
            view.add_callback('new frame', self.update_position)

    def update_position(self):
        m = self.model
        fr = self.frame
        if fr >= self.frames:
            m.position = self.pos1
            self.view.remove_callback('new frame', self.update_position)
        else:
            f = fr / self.frames
            from ..geometry import translation, rotation
            m.position = translation(f*(self.c1-self.c0)) * rotation(self.axis, f*self.angle, self.c0) * self.pos0
            self.frame += 1

def register_crosslink_command():
    from . import cli
    from .color import ColorArg
    desc = cli.CmdDesc(optional = [('pbgroups', cli.PseudobondGroupsArg)],
                       keyword = [('color', ColorArg),
                                  ('radius', cli.FloatArg),
                                  ('minimize', cli.AtomicStructuresArg),
                                  ('iterations', cli.IntArg),
                                  ('frames', cli.IntArg),
                              ])
    cli.register('crosslink', desc, crosslink)
