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

def crosslinks(session, pbgroups = None, color = None, radius = None,
               plot = None, minimize = None, iterations = 10, frames = None):
    '''
    Move atomic models to minimize crosslink lengths.

    Parameters
    ----------
    pbgroups : PseudobondGroups or None
      Pseudobond groups containing crosslinks.  If None then all pseudbond groups are used.
    color : Color
      Set the pseudobonds to this color
    radius : float
      Set pseudobond cylinder radius.
    plot : bool
      Whether to show a 2d graph of chains with crosslinks between them.
    minimize : bool
      Move each atomic structure model rigidly to minimize the sum of squares of link distances
      to other models.  Each model is moved one time.  This does not produce minimum sum of squares
      of all links, but multiple iterations converge to that result.
    iterations : int
      Minimize the sequence of atomic structures this many times.
    frames : int
      If minimize is true then move the atomic structures gradually to their minimized positions
      over this many frames.
    '''
    if pbgroups is None:
        from chimerax.core import atomic
        pbgroups = atomic.all_pseudobond_groups(session.models)

    if len(pbgroups) == 0:
        from chimerax.core.errors import UserError        
        raise UserError('No pseudobond groups specified.')

    from chimerax.core.atomic import concatenate
    pbonds = concatenate([pbg.pseudobonds for pbg in pbgroups])

    if color:
        rgba = color.uint8x4()
        for pb in pbonds:
            pb.color = rgba

    if radius:
        for pb in pbonds:
            pb.radius = radius

    if plot:
        from .chainplot import CrosslinksPlot, chains_and_edges
        from chimerax.core.atomic import concatenate, Pseudobonds
        pbonds = concatenate([pbg.pseudobonds for pbg in pbgroups], Pseudobonds)
        cnodes, edges = chains_and_edges(pbonds)
        CrosslinksPlot(session, cnodes, edges)

    if minimize:
        minimize_link_lengths(minimize, pbonds, iterations, frames, session)

def minimize_link_lengths(mols, pbonds, iterations, frames, session):
    if len(mols) == 0:
        from chimerax.core.errors import UserError        
        raise UserError('No structures specified for minimizing crosslinks.')
    mol_links, mol_pbonds = links_by_molecule(pbonds, mols)
    if len(mol_links) == 0:
        from chimerax.core.errors import UserError        
        raise UserError('No pseudobonds to minimize for specified molecules.')
    if len(mols) == 1:
        iterations = min(1,iterations)
    if not frames is None:
        pos0 = dict((m,m.position) for m in mols)
    from numpy import array, float64
    from chimerax.core.geometry import align_points
    for i in range(iterations):
        for m in mols:
            if m in mol_links:
                atom_pairs = mol_links[m]
                moving = array([a1.scene_coord for a1,a2 in atom_pairs], float64)
                fixed = array([a2.scene_coord for a1,a2 in atom_pairs], float64)
                tf, rms = align_points(moving, fixed)
                m.position = tf * m.position

    lengths = [pb.length for pb in mol_pbonds]
    lengths.sort(reverse = True)
    lentext = ', '.join('%.1f' % d for d in lengths)
    session.logger.info('%d crosslinks, lengths: %s' % (len(mol_pbonds), lentext))

    if not frames is None:
        for m in mols:
            interpolate_position(m, pos0[m], m.position, frames, session.triggers)

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

    def __init__(self, model, pos0, pos1, frames, triggers):
        self.model = model
        self.pos0 = pos0
        self.pos1 = pos1
        self.frames = frames
        self.frame = 1
        self.ses_triggers = triggers

        b = model.bounds()
        if b is None:
            model.position = pos1
        else:
            center = 0.5*(b.xyz_min + b.xyz_max)
            self.c0, self.c1 = pos0*center, pos1*center
            self.axis, self.angle = (pos1*pos0.inverse()).rotation_axis_and_angle()
            triggers.add_handler('new frame', self.update_position)

    def update_position(self, *_):
        m = self.model
        fr = self.frame
        if fr >= self.frames:
            m.position = self.pos1
            from chimerax.core.triggerset import DEREGISTER
            return DEREGISTER
        else:
            f = fr / self.frames
            from chimerax.core.geometry import translation, rotation
            m.position = translation(f*(self.c1-self.c0)) * rotation(self.axis, f*self.angle, self.c0) * self.pos0
            self.frame += 1

def register_command():
    from chimerax.core.commands import register, CmdDesc, ColorArg, FloatArg, IntArg, PseudobondGroupsArg, StructuresArg, NoArg
    desc = CmdDesc(optional = [('pbgroups', PseudobondGroupsArg)],
                       keyword = [('color', ColorArg),
                                  ('radius', FloatArg),
                                  ('plot', NoArg),
                                  ('minimize', StructuresArg),
                                  ('iterations', IntArg),
                                  ('frames', IntArg),
                              ])
    register('crosslinks', desc, crosslinks)
