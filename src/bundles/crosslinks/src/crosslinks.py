# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2022 Regents of the University of California. All rights reserved.
# This software is provided pursuant to the ChimeraX license agreement, which
# covers academic and commercial uses. For more information, see
# <http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
#
# This file is part of the ChimeraX library. You can also redistribute and/or
# modify it under the GNU Lesser General Public License version 2.1 as
# published by the Free Software Foundation. For more details, see
# <https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html>
#
# This file is distributed WITHOUT ANY WARRANTY; without even the implied
# warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. This notice
# must be embedded in or attached to all copies, including partial copies, of
# the software or any revisions or derivations thereof.
# === UCSF ChimeraX Copyright ===

def crosslinks(session, pbonds, color = None, radius = None, dashes = None):
    '''
    Set crosslink colors and radii.

    Parameters
    ----------
    pbonds : Pseudobonds
      Crosslinks to display or minimize.
    color : Color
      Set the pseudobonds to this color
    radius : float
      Set pseudobond cylinder radius.
    dashes : int
      Number of dashes shown for pseudobonds.  Applies to whole pseudobond groups.
    '''

    if len(pbonds) == 0:
        from chimerax.core.errors import UserError        
        raise UserError('No pseudobonds specified.')

    if color:
        rgba = color.uint8x4()
        for pb in pbonds:
            pb.color = rgba

    if radius:
        for pb in pbonds:
            pb.radius = radius

    if dashes is not None:
        for pbg in pbonds.groups.unique():
            pbg.dashes = dashes

def crosslinks_network(session, pbonds):
    '''
    Display a graph of chains as nodes and edges labeled with number
    of crosslinks between chains.

    Parameters
    ----------
    pbonds : Pseudobonds
      Crosslinks to show in graph
    '''

    if len(pbonds) == 0:
        from chimerax.core.errors import UserError        
        raise UserError('No pseudobonds specified.')

    from .chainplot import CrosslinksPlot, chains_and_edges
    cnodes, edges = chains_and_edges(pbonds)
    CrosslinksPlot(session, cnodes, edges)

def crosslinks_histogram(session, pbonds, coordsets = None, bins = 50,
                         max_length = None, min_length = None, height = None):
    '''
    Show histogram of crosslink lengths.

    Parameters
    ----------
    pbonds : Pseudobonds
      Crosslinks to show in histogram
    coordsets : AtomicStructure
      If a single pseudobond is specified plot a histogram of its
      length across the specified coordinate sets is produced.
    '''

    if len(pbonds) == 0:
        from chimerax.core.errors import UserError        
        raise UserError('No pseudobonds specified.')

    if coordsets:
        if len(pbonds) == 1:
            from .lengths import EnsemblePlot
            plot = EnsemblePlot(session, pbonds[0], coordsets,
                                bins=bins, max_length=max_length, min_length=min_length, height=height)
        else:
            from chimerax.core.errors import UserError        
            raise UserError('Plotting coordset lengths requires exactly one crosslink, got %d.' % len(pbonds))
    else:
        from .lengths import LengthsPlot
        plot = LengthsPlot(session, pbonds,
                           bins=bins, max_length=max_length, min_length=min_length, height=height)
    return plot

def crosslinks_minimize(session, pbonds, move_models = None, iterations = 10, frames = None):
    '''
    Move each atomic structure model rigidly to minimize the sum of squares of link distances
    to other models.  Each model is moved one time.  This does not produce minimum sum of squares
    of all links, but multiple iterations converge to that result.

    Parameters
    ----------
    pbonds : Pseudobonds
      Crosslinks to display or minimize.
    move_models : list of Structures or None
      Which structures to move. If None then move all that are attached to specified crosslinks.
    iterations : int
      Minimize the sequence of atomic structures this many times.
    frames : int
      If minimize is true then move the atomic structures gradually to their minimized positions
      over this many frames.
    '''

    if len(pbonds) == 0:
        from chimerax.core.errors import UserError        
        raise UserError('No pseudobonds specified.')

    if move_models is None:
        move_models = pbonds.unique_structures
    minimize_link_lengths(move_models, pbonds, iterations, frames, session)

def minimize_link_lengths(mols, pbonds, iterations, frames, session):
    if len(mols) == 0:
        from chimerax.core.errors import UserError        
        raise UserError('No structures specified for minimizing crosslinks.')
    mol_links, mol_pbonds = links_by_molecule(pbonds, mols)
    if len(mol_links) == 0:
        from chimerax.core.errors import UserError        
        raise UserError('No pseudobonds between molecules.')
    if len(mols) == 1:
        iterations = min(1,iterations)
    if not frames is None:
        pos0 = dict((m,m.position) for m in mols)
    from numpy import array, float64
    from chimerax.geometry import align_points
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
            center = model.scene_position.inverse() * b.center() # Center in model coords
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
            from chimerax.geometry import translation, rotation
            m.position = translation(f*(self.c1-self.c0)) * rotation(self.axis, f*self.angle, self.c0) * self.pos0
            self.frame += 1

def register_command(logger):
    from chimerax.core.commands import register, CmdDesc, ColorArg, FloatArg, IntArg
    from chimerax.atomic import StructureArg, StructuresArg, PseudobondsArg
    desc = CmdDesc(required = [('pbonds', PseudobondsArg)],
                   keyword = [('color', ColorArg),
                              ('radius', FloatArg),
                              ('dashes', IntArg)],
                   synopsis = 'Set crosslink colors and radii')
    register('crosslinks', desc, crosslinks, logger=logger)

    desc = CmdDesc(required = [('pbonds', PseudobondsArg)],
                   synopsis = 'Plot graph of crosslink connections between chains')
    register('crosslinks network', desc, crosslinks_network, logger=logger)

    desc = CmdDesc(required = [('pbonds', PseudobondsArg)],
                   keyword = [('coordsets', StructureArg),
                              ('bins', IntArg),
                              ('max_length', FloatArg),
                              ('min_length', FloatArg),
                              ('height', FloatArg)],
                   synopsis = 'Show histogram of crosslink lengths')
    register('crosslinks histogram', desc, crosslinks_histogram, logger=logger)

    desc = CmdDesc(required = [('pbonds', PseudobondsArg)],
                   keyword = [('move_models', StructuresArg),
                              ('iterations', IntArg),
                              ('frames', IntArg),],
                   synopsis = 'Minimize crosslink lengths')
    register('crosslinks minimize', desc, crosslinks_minimize, logger=logger)
