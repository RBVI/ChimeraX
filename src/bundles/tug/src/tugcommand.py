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

def tug(session, atoms, to_atoms, force_constant=1000, cutoff=10,
        temperature=100, error_tolerance = 0.001,
        steps=50, frames=50, finish=False):
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
    	Strength of tugging force, Newtons per meter, default 1000.
    cutoff : float
        Non-bonded force distance cutoff in Angstroms, default 10.
    temperature : float
        Simulation temperature in Kelvin, default 100.
    error_tolerance : float
        Langevin integrator tolerance for OpenMM, default 0.001.
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

    from .tugatoms import StructureTugger, ForceFieldError
    try:
        tugger = StructureTugger(us[0], force_constant = force_constant,
                                 cutoff = cutoff, temperature = temperature,
                                 tolerance = error_tolerance, steps = steps)
    except ForceFieldError as e:
        # Structure could not be parameterized.
        from chimerax.core.errors import UserError
        raise UserError(str(e))
    
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
                              ('cutoff', FloatArg),
                              ('temperature', FloatArg),
                              ('error_tolerance', FloatArg),
                              ('steps', IntArg),
                              ('frames', IntArg),
                              ('finish', BoolArg)],
                   required_arguments = ['to_atoms'],
                   synopsis='Tug atoms while running molecular dynamics')
    register('tug', desc, tug, logger=logger)
