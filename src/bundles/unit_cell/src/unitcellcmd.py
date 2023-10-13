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

# -----------------------------------------------------------------------------------
# Create molecule copies to fill crystal unit cell.
#
def unitcell(session, structures = None, cells = (1,1,1), offset = (0,0,0), origin = (0,0,0),
             sym_from_file = True, spacegroup = True, ncs = True, pack = True,
             outline = False):
    '''
    Make copies of an atomic model to fill out the crystal unit cell.

    Parameters
    ----------
    structures : list of AtomicStructure
        One or more structures to operate on.
    cells : 3 positive integers
        Number of unit cells to fill along 3 axes.  Default (1,1,1)
    offset : 3 integers
        Position of the unit cell in the lattice.  If more than one unit cell
        is filled then this is the position of the lower left front unit cell.
        Default (0,0,0).
    origin : 3 floats
        Fractional coordinates of the lower left front corner of the unit cell.
        This allows shifting the outline box and controls the box that the copies
        are packed into.  Default (0,0,0).
    sym_from_file : bool
        Whether to use crystal symmetry matrices specified in mmCIF and PDB files.
        Default true.
    spacegroup : bool
        Whether to determine the crystal symmetries from the space group specified
        in the file.  This will only be used if explicit symmetry matrices are not
        in the file or sym_from_file is false.  Default true.
    ncs : bool
        Whether to use non-crystallography symmetry matrices from the file.
        Default true.
    pack : bool
        Whether to place each copy so that its center is within the unit cell box.
        This translates copies by whole unit cell displacements if the copy would
        lie outside the box if directly using the file symmetry matrices.
        Default true.
    outline : bool
        Whether to show an outline box around a single unit cell.  Default false.
    '''

    if structures is None:
        structures = _all_structures(session)

    if len(structures) == 0:
        from chimerax.core.errors import UserError
        raise UserError('No structures specified')

    from . import unitcell
    for m in structures:
        tflist = unitcell.transforms(m, cells, offset, origin, sym_from_file,
                                     spacegroup, ncs, pack)
        group_model = unitcell.copies_group_model(m)
        unitcell.place_molecule_copies(m, group_model, tflist)
        unitcell.remove_extra_copies(m, group_model, len(tflist))
        if outline:
            unitcell.show_outline_model(m, origin)

# -----------------------------------------------------------------------------------
#
def unitcell_outline(session, structures = None, origin = (0,0,0), close = False):
    if structures is None:
        structures = _all_structures(session)

    from . import unitcell
    for m in structures:
        if close:
            outline_model = unitcell.outline_model(m)
            if outline_model:
                session.models.close([outline_model])
        else:
            unitcell.show_outline_model(m, origin)

# -----------------------------------------------------------------------------------
#
def unitcell_info(session, structures = None):
    if structures is None:
        structures = _all_structures(session)

    infos = []
    from . import unitcell
    for m in structures:
        info = unitcell.unit_cell_info(m)
        infos.append('%s (#%s) unit cell\n%s' % (m.name, m.id_string, info))

    msg = '\n\n'.join(infos)
    session.logger.info(msg)
    
# -----------------------------------------------------------------------------------
#
def unitcell_delete(session, structures = None, outline = True):
    if structures is None:
        structures = _all_structures(session)
        
    from . import unitcell
    for m in structures:
        gm = unitcell.copies_group_model(m, create = False) 
        if gm:
            session.models.close([gm])

# -----------------------------------------------------------------------------------
#
def _all_structures(session):
    from chimerax import atomic
    structures = [s for s in atomic.all_atomic_structures(session)
                  if not hasattr(s, '_unit_cell_copy')]
    return structures

# -----------------------------------------------------------------------------------
#
def register_unitcell_command(logger):

    from chimerax.core.commands import CmdDesc, register
    from chimerax.core.commands import Int3Arg, Float3Arg, BoolArg, NoArg
    from chimerax.atomic import AtomicStructuresArg

    desc = CmdDesc(
        optional = [('structures', AtomicStructuresArg)],
        keyword = [('cells', Int3Arg),
                   ('offset', Int3Arg),
                   ('origin', Float3Arg),
                   ('sym_from_file', BoolArg),
                   ('spacegroup', BoolArg),
                   ('ncs', BoolArg),
                   ('pack', BoolArg),
                   ('outline', BoolArg)],
        synopsis = 'Copy atomic structure to fill crystal unit cell'
    )
    register('unitcell', desc, unitcell, logger=logger)

    desc = CmdDesc(optional = [('structures', AtomicStructuresArg)],
                   keyword = [('origin', Float3Arg),
                              ('close', NoArg)],
                   synopsis = 'Show unit cell outline box')
    register('unitcell outline', desc, unitcell_outline, logger=logger)

    desc = CmdDesc(optional = [('structures', AtomicStructuresArg)],
                   synopsis = 'Report unit cell information for atomic structure')
    register('unitcell info', desc, unitcell_info, logger=logger)

    desc = CmdDesc(optional = [('structures', AtomicStructuresArg)],
                   keyword = [('outline', BoolArg)],
                   synopsis = 'Close atomic structure copies made with unitcell command')
    register('unitcell delete', desc, unitcell_delete, logger=logger)
