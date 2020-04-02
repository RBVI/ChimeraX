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

from chimerax.atomic import Element

def place_helium(structure, res_name, position=None):
    '''If position is None, place in the center of view'''
    max_existing = 0
    for r in structure.residues:
        if r.chain_id == "het" and r.number > max_existing:
            max_existing = r.number
    res = structure.new_residue(res_name, "het", max_existing+1)
    if position is None:
        if len(structure.session.models) == 0:
            from numpy import array
            position = array([0.0,0.0,0.0])
        else:
            view = structure.session.view
            n, f = view.near_far_distances(view.camera, None)
            position = view.camera.position.origin() + (n+f) * view.camera.view_direction() / 2
    from chimerax.atomic.struct_edit import add_atom
    helium = Element.get_element("He")
    a = add_atom("He", helium, res, position)
    from chimerax.atomic.colors import element_color
    a.color = element_color(helium.number)
    a.draw_mode = a.BALL_STYLE
    return a

class PeptideError(ValueError):
    pass

def place_peptide(structure, sequence, phi_psis, *, position=None, rot_lib=None, chain_id=None):
    """
    Place a peptide sequence.

    *structure* is an AtomicStructure to add the peptide to.

    *sequence* contains the sequence as a string.

    *phi_psis* is a list of phi/psi tuples, one per residue.

    *position* is either a numpy xyz array or None.  If None, the peptide is positioned at the
    center of the view.

    *rot_lib* is the name of the rotamer library to use to position the side chains.
    session.rotamers.library_names lists installed rotamer library names.  If None, a default
    library will be used.

    *chain_id* is the desired chain ID for the peptide.  If None, earliest alphabetical chain ID
    not already in use in the structure will be used (upper case taking precedence over lower
    case).
    """

    if not sequence:
        raise PeptideError("No sequence supplied")
    sequence = sequence.upper()
    if not sequence.isupper():
        raise PeptideError("Sequence contains non-alphabetic characters")
    from chimerax.atomic import Sequence
    for c in sequence:
        try:
            r3 = Sequence.protein1to3[c]
        except KeyError:
            raise PeptideError("Unrecognized protein 1-letter code: %s" % c)
        if r3[-1] == 'X':
            raise PeptideError("Peptide sequence cannot contain ambiguity codes (i.e. '%s')" % c)

    if len(sequence) != len(phi_psis):
        raise PeptideError("Number of phi/psis not equal to sequence length")

    session = structure.session

    open_models = session.models[:]
    if len(open_models) == 0:
        need_focus = True
    elif len(open_models) == 1:
        if open_models[0] == structure:
            need_focus = not structure.atoms
        else:
            need_focus = False
    else:
        need_focus = False

    if position is None:
        position = session.main_view.center_of_rotation

    #TODO: actually add the peptide
