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

def distance(session, atoms):
    '''
    Show/report distance between two atoms.
    '''
    grp = session.pb_manager.get_group("distances", create=False)
    if not grp:
        # create group and add to DistMonitor
        grp = session.pb_manager.get_group("distances")
        session.models.add([grp])
        session.pb_dist_monitor.add_group(grp)
    a1, a2 = atoms
    pb = grp.new_pseudobond(a1, a2)
    session.logger.info(("Distance between %s and %s: " + session.pb_dist_monitor.distance_format)
        % (a1, a2.__str__(relative_to=a1), pb.length))

def xdistance(session, pbonds=None):
    pbg = session.pb_manager.get_group("distances", create=False)
    if not pbg:
        return
    dist_pbonds = pbonds.with_group(pbg) if pbonds != None else None
    if pbonds == None or len(dist_pbonds) == pbg.num_pseudobonds:
        session.models.close([pbg])
        return
    for pb in dist_pbonds:
        pbg.delete_pseudobond(pb)

def register_command(session):
    from . import CmdDesc, register, AtomsArg, AnnotationError, PseudobondsArg, Or, EmptyArg
    # eventually this will handle more than just atoms, but for now...
    class AtomPairArg(AtomsArg):
        name = "an atom-pair specifier"

        @classmethod
        def parse(cls, text, session):
            atoms, text, rest = super().parse(text, session)
            if len(atoms) != 2:
                raise AnnotationError("Expected two atoms to be specified (%d specified)"
                    % len(atoms))
            return atoms, text, rest
    d_desc = CmdDesc(
        required = [('atoms', AtomPairArg)],
        synopsis = 'show/report distance')
    register('distance', d_desc, distance, logger=session.logger)
    xd_desc = CmdDesc(
        required = [('pbonds', Or(PseudobondsArg,EmptyArg))],
        synopsis = 'remove distance monitors')
    register('~distance', xd_desc, xdistance, logger=session.logger)
