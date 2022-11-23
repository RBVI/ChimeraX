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

# -----------------------------------------------------------------------------
#
def esmfold_contacts(session, residues, to_residues = None, distance = 3,
                     flip = False, palette = None, range = None, radius = 0.2, dashes = 1,
                     name = 'PAE Contacts', replace = True, output_file = None):
    '''
    Create pseudobonds between close residues of an ESMFold structure
    colored by the predicted aligned error value.  The paecontacts colormap
    is used ranging from blue for low error to red for high error.
    Pseudobonds are drawn between CA atoms.
    '''
    from chimerax.alphafold.contacts import alphafold_contacts
    return alphafold_contacts(session, residues, to_residues = to_residues, distance = distance,
                              flip = flip, palette = palette, range = range, radius = radius, dashes = dashes,
                              name = name, replace = replace, output_file = output_file, method = 'esmfold')
    
# -----------------------------------------------------------------------------
#
def register_esmfold_contacts_command(logger):
    from chimerax.alphafold import contacts
    desc = contacts.contacts_command_description()
    from chimerax.core.commands import register
    register('esmfold contacts', desc, esmfold_contacts, logger=logger)
