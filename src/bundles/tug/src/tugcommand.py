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

def tug(session, atoms, to_atoms, force_constant=1000, steps=50, frames=50, finish=False):
    '''
    Run molecular dynamics on a structure tugging some atoms.
    This is a command version of the tug atom mouse mode.

    Parameters
    ----------
    atoms : Atoms
        Which atoms to tug.
    to_atoms : Atoms
        Tug atoms toward current location of these atoms.
    force_constant : float
    	Strength of tugging force, Newtons per meter.
    steps : integer
        How many time steps to take each graphics frame.
    frames : integer
        How many graphics frames to tug for.
    '''
    us = atoms.unique_structures
    if len(us) > 1:
        from chimerax.core.errors import UserError
        raise UserError('Can only run tug command on a single structure, got %d' % len(us))
    if len(to_atoms) != len(atoms):
        from chimerax.core.errors import UserError
        raise UserError('For %d tugged atoms expected %d destination atoms, got %d'
                        % (len(atoms), len(atoms), len(to_atoms)))

    from .tugatoms import StructureTugger
    tugger = StructureTugger(us[0])
    tugger._force_constant = force_constant
    tugger._sim_steps = steps
    tugger.tug_atoms(atoms)
    points = to_atoms.scene_coords
    
    def simulation_frame(session, frame, tugger=tugger, points=points):
        tugger.tug_to_positions(points)
        
    from chimerax.core.commands.motion import CallForNFrames
    CallForNFrames(simulation_frame, frames, session)

    if finish:
        from chimerax.std_commands.wait import wait
        wait(session, frames)

    return tugger
    
def register_tug_command(logger):
    from chimerax.core.commands import register, CmdDesc, CenterArg, FloatArg, IntArg, BoolArg
    from chimerax.atomic import AtomsArg
    desc = CmdDesc(required = [('atoms', AtomsArg)],
                   keyword = [('to_atoms', AtomsArg),
                              ('force_constant', FloatArg),
                              ('steps', IntArg),
                              ('frames', IntArg),
                              ('finish', BoolArg)],
                   required_arguments = ['to_atoms'],
                   synopsis='Tug an atom while running molecular dynamics')
    register('tug', desc, tug, logger=logger)
