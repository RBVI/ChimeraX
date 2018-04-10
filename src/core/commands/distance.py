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

def distance(session, atoms, *, color=None, dashes=None, radius=None):
    '''
    Show/report distance between two atoms.
    '''
    grp = session.pb_manager.get_group("distances", create=False)
    if not grp:
        # create group and add to DistMonitor
        grp = session.pb_manager.get_group("distances")
        if color is not None:
            grp.color = color.uint8x4()
        if radius is not None:
            grp.radius = radius
        session.models.add([grp])
        session.pb_dist_monitor.add_group(grp)
    a1, a2 = atoms
    # might just be changing color/radius, so look for existing pseudobond
    update_label_color = False
    for pb in grp.pseudobonds:
        pa1, pa2 = pb.atoms
        if (pa1 == a1 and pa2 == a2) or (pa1 == a2 and pa2 == a1):
            update_label_color = True
            break
    else:
        pb = grp.new_pseudobond(a1, a2)
    if color is not None:
        pb.color = color.uint8x4()
        if update_label_color:
            from chimerax.label.label3d import labels_model, PseudobondLabel
            lm = labels_model(grp, create=False)
            if lm:
                lm.add_labels([pb], PseudobondLabel, session.main_view,
                    settings={ 'color': pb.color })
    if dashes is not None:
        grp.dashes = dashes
    if radius is not None:
        pb.radius = radius
    session.logger.info(("Distance between %s and %s: " + session.pb_dist_monitor.distance_format)
        % (a1, a2.string(relative_to=a1), pb.length))

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

def distance_format(session, *, decimal_places=None, symbol=None, save=False):
    session.pb_dist_monitor.set_distance_format_params(decimal_places=decimal_places,
        show_units=symbol, save=save)

def register_command(session):
    from . import CmdDesc, register, AtomsArg, AnnotationError, PseudobondsArg, Or, EmptyArg, \
        ColorArg, NonNegativeIntArg, FloatArg, BoolArg
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
        keyword = [('color', ColorArg), ('dashes', NonNegativeIntArg), ('radius', FloatArg)],
        synopsis = 'show/report distance')
    register('distance', d_desc, distance, logger=session.logger)
    xd_desc = CmdDesc(
        required = [('pbonds', Or(PseudobondsArg,EmptyArg))],
        synopsis = 'remove distance monitors')
    register('~distance', xd_desc, xdistance, logger=session.logger)
    df_desc = CmdDesc(
        keyword = [('decimal_places', NonNegativeIntArg), ('symbol', BoolArg), ('save', BoolArg)],
        synopsis = 'set distance formatting')
    register('distance format', df_desc, distance_format, logger=session.logger)
