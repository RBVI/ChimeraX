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

# -----------------------------------------------------------------------------
# Command to show crystal contacts.
#
def crystalcontacts(session, structures = None, distance = 3.0,
                    residue_info = False, buried_areas = False, probe_radius = 1.4,
                    intra_biounit = True,
                    copies = True, rainbow = True, schematic = False):
    '''
    Make contacting copies of an atomic model of a crystal asymmetric unit.

    Parameters
    ----------
    structures : list of AtomicStructure
        One or more structures to operate on.
    distance : float
        Maximum distance between asymmetric units (atom to atom).  Default 3 Angstroms.
    residue_info : bool
        Whether to log residue contact information. Default false.
    buried_areas : bool
        Compute buried solvent accessible surface areas per residue and report
        in Log if residue_info is true.  Default false.
    probe_radius : float
        Probe radius for computing solvent accessible areas.  Default 1.4 Angstroms.
    intra_biounit : bool
        Whether contacts between subunits in the same biological unit should be included
        in the results. The biological unit is defined by BIOMT matrices in the input PDB file.
        Default true.
    copies : bool
        Whether copies of the contacting asymmetric units should be made.  Default true.
    rainbow : bool
        Whether to give different colors to each contacting copy.  If false then the
        copies are colored the same as the original.  Default true.
    schematic : bool
        Whether a schematic of the contacting units shown as balls is shown.  Default false.
    '''

    if structures is None:
        structures = _all_structures(session)

    if len(structures) == 0:
        from chimerax.core.errors import UserError
        raise UserError('No structures specified')

    from .crystal import show_crystal_contacts
    for m in structures:
        show_crystal_contacts(m, distance,
                              make_copies = copies,
                              rainbow = rainbow,
                              schematic = schematic,
                              residue_info = residue_info,
                              buried_areas = buried_areas,
                              probe_radius = probe_radius,
                              intra_biounit = intra_biounit)
    
# -----------------------------------------------------------------------------------
#
def crystalcontacts_delete(session, structures = None):
    if structures is None:
        structures = _all_structures(session)
        
    from . import crystal
    for m in structures:
        gm = crystal.copies_group_model(m, create = False) 
        if gm:
            session.models.close([gm])
        sm = crystal.schematic_model(m)
        if sm:
            session.models.close([sm])

# -----------------------------------------------------------------------------------
#
def _all_structures(session):
    from chimerax import atomic
    structures = [s for s in atomic.all_atomic_structures(session)
                  if not hasattr(s, '_crystal_contacts_copy')]
    return structures

# -----------------------------------------------------------------------------------
#
def register_crystalcontacts_command(logger):

    from chimerax.core.commands import CmdDesc, register
    from chimerax.core.commands import FloatArg, BoolArg
    from chimerax.atomic import AtomicStructuresArg

    desc = CmdDesc(
        optional = [('structures', AtomicStructuresArg)],
        keyword = [('distance', FloatArg),
                   ('residue_info', BoolArg),
                   ('buried_areas', BoolArg),
                   ('probe_radius', FloatArg),
                   ('intra_biounit', BoolArg),
                   ('copies', BoolArg),
                   ('rainbow', BoolArg),
                   ('schematic', BoolArg)],
        synopsis = 'Find contacting copies of crystal asymmetric unit'
    )
    register('crystalcontacts', desc, crystalcontacts, logger=logger)

    desc = CmdDesc(optional = [('structures', AtomicStructuresArg)],
                   synopsis = 'Close atomic structure copies made with crystalcontacts command')
    register('crystalcontacts delete', desc, crystalcontacts_delete, logger=logger)
