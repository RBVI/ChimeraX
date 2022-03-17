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

def tug(session, atom, to_point, force_constant=10000, steps=50, frames=50):
    '''
    Run molecular dynamics on a structure tugging an atom.
    This is a command version of the tug atom mouse mode.

    Parameters
    ----------
    atom : Atom
        Which atom to tug.
    to_point : Center
        Point atom should be tugged toward
    force_constant : float
    	Strength of tugging force, Joules per meter.
    steps : integer
        How many time steps to take each graphics frame.
    frames : integer
        How many graphics frames to tug for.
    '''
    from .tugatoms import StructureTugger
    tugger = StructureTugger(atom.structure)
    tugger._force_constant = force_constant
    tugger._sim_steps = steps
    tugger.tug_atom(atom)
    target = to_point.scene_coordinates()
    
    def simulation_frame(session, frame, tugger=tugger, point=target):
        delta = point - tugger.atom.scene_coord
        tugger.tug_displacement(delta)
        
    from chimerax.core.commands.motion import CallForNFrames
    CallForNFrames(simulation_frame, frames, session)

    return tugger
    
def register_tug_command(logger):
    from chimerax.core.commands import register, CmdDesc, CenterArg, FloatArg, IntArg
    from chimerax.atomic import AtomArg
    desc = CmdDesc(required = [('atom', AtomArg)],
                   keyword = [('to_point', CenterArg),
                              ('force_constant', FloatArg),
                              ('steps', IntArg),
                              ('frames', IntArg)],
                   required_arguments = ['to_point'],
                   synopsis='Tug an atom while running molecular dynamics')
    register('tug', desc, tug, logger=logger)
