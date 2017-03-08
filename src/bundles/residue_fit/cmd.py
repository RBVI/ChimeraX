# vim: set expandtab ts=4 sw=4:

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

def resfit(session, atoms, map = None, residue_range = (-2,1),
           motion_frames = 50, pause_frames = 50, movie_framerate = 25):
    '''Display fit of each residue in a density map.

    Parameters
    ----------
    atoms : Atoms
      Atoms from one chain or part of a chain.
    map : Volume
      Density map to show near each residue.
    '''

    from chimerax.core.commands import AnnotationError
    if map is None:
        raise AnnotationError('Require "map" option: resfit #1 map #2')

    cids = atoms.unique_chain_ids
    if len(cids) != 1:
        raise AnnotationError('Atoms must belong to one chain, got %d chains %s'
                              % (len(cids), ', '.join(cids)))

    res = atoms.unique_residues
    bbres = residues_with_backbone(res)
    if len(bbres) == 0:
        raise AnnotationError('None of %d specified residues have backbone atoms "N", "CA" and "C"' % len(res))
    
    from . import tool
    tool.ResidueFit(session, "Residue Fit", bbres, map, residue_range = residue_range,
                   motion_frames = motion_frames, pause_frames = pause_frames,
                   movie_framerate = movie_framerate)
    

def residues_with_backbone(residues):
    rb = []
    for i,r in enumerate(residues):
        anames = r.atoms.names
        if 'N' in anames and 'CA' in anames and 'C' in anames:
            rb.append(i)
    return residues.filter(rb)

def register_resfit_command(logger):
    from chimerax.core.commands import CmdDesc, register, AtomsArg, IntArg, Int2Arg
    from chimerax.core.map import MapArg
    desc = CmdDesc(required = [('atoms', AtomsArg)],
                   keyword = [('map', MapArg),
                              ('residue_range', Int2Arg),
                              ('motion_frames', IntArg),
                              ('pause_frames', IntArg),
                              ('movie_framerate', IntArg)],
                   synopsis = 'Display slider to show fit of each residue in density map')
    register('resfit', desc, resfit, logger=logger)
