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

def style(session, objects=None, atom_style=None, dashes=None, ring_fill=None):
    '''Set the atom and bond display styles.

    Parameters
    ----------
    objects : Objects
        Change the style of these atoms, bonds and pseudobonds.
        If not specified then all atoms are changed.
    atom_style : "sphere", "ball" or "stick"
        Controls how atoms and bonds are depicted.
    dashes : int
        Optional number of dashes shown for pseudobonds.
    ring_fill : Optional "thick", "thin", or "off".
    '''
    if objects is None:
        from chimerax.core.commands import all_objects
        objects = all_objects(session)
    atoms = objects.atoms

    from chimerax.core.commands import plural_form
    from chimerax.core.undo import UndoState
    undo_state = UndoState("style")
    what = []
    if atom_style is not None:
        from chimerax.atomic import Atom
        s = {
            'sphere': Atom.SPHERE_STYLE,
            'ball': Atom.BALL_STYLE,
            'stick': Atom.STICK_STYLE,
        }[atom_style.lower()]
        undo_state.add(atoms, "draw_modes", atoms.draw_modes, s)
        atoms.draw_modes = s
        what.append('%d %s' % (len(atoms), plural_form(atoms, 'atom style')))

    if dashes is not None:
        pbgs = objects.pseudobonds.unique_groups
        for pbg in pbgs:
            undo_state.add(pbg, "dashes", pbg.dashes, dashes)
            pbg.dashes = dashes
        what.append('%d %s' % (len(pbgs), plural_form(pbgs, 'pseudobond dash')))

    if ring_fill is not None:
        atoms = objects.atoms
        res = atoms.unique_residues
        if ring_fill == 'on':
            undo_state.add(res, "ring_displays", res.ring_displays, True)
            res.ring_displays = True
        elif ring_fill == 'off':
            undo_state.add(res, "ring_displays", res.ring_displays, False)
            res.ring_displays = False
        elif ring_fill == 'thin':
            undo_state.add(res, "ring_displays", res.ring_displays, True)
            undo_state.add(res, "thin_rings", res.thin_rings, True)
            res.ring_displays = True
            res.thin_rings = True
        elif ring_fill == 'thick':
            undo_state.add(res, "ring_displays", res.ring_displays, True)
            undo_state.add(res, "thin_rings", res.thin_rings, False)
            res.ring_displays = True
            res.thin_rings = False
        what.append('%d %s' % (len(res), plural_form(res, 'residue ring style')))

    if what:
        msg = 'Changed %s' % ', '.join(what)
        log = session.logger
        log.status(msg)
        log.info(msg)

    session.undo.register(undo_state)

# -----------------------------------------------------------------------------
#
def register_command(logger):
    from chimerax.core.commands import register, CmdDesc, ObjectsArg, EmptyArg, EnumOf, Or, IntArg
    desc = CmdDesc(required = [('objects', Or(ObjectsArg, EmptyArg)),
                               ('atom_style', Or(EnumOf(('sphere', 'ball', 'stick')), EmptyArg))],
                   keyword = [('dashes', IntArg),
                              ('ring_fill', EnumOf(['on', 'off', 'thick', 'thin']))],
                   synopsis='change atom and bond depiction')
    register('style', desc, style, logger=logger)
