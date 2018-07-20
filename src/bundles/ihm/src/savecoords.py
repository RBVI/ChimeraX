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

def save_binary_coordinates(session, filename, models):
    if models is None:
        from chimerax.core.errors import UserError
        raise UserError('Must specify models to save coordinates')
    from chimerax.atomic import Structure
    mlist = [m for m in models if isinstance(m, Structure)]
    if len(mlist) == 0:
        from chimerax.core.errors import UserError
        raise UserError('No structures to save')
    na = mlist[0].num_atoms
    for m in mlist[1:]:
        if m.num_atoms != na:
            from chimerax.core.errors import UserError
            raise UserError('Saving coordinates requires all structures have the same number of atoms, got %s'
                            % ', '.join('%d' % m.num_atoms for m in mlist))
    from chimerax.ihm import coordsets
    coordsets.write_coordinate_sets(filename, mlist)
