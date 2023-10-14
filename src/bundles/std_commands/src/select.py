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

def select(session, objects=None, residues=False, minimum_length=None, maximum_length=None,
        sequence=None):
    '''Select specified objects.

    Parameters
    ----------
    objects : Objects
      Replace the current selection with the specified objects (typically atoms).
      If no objects are specified then everything is selected.
    residues : bool
      Extend atoms that are selected to containing residues if true (default false).
    minimum_length : float or None
      Exclude pseudobonds shorter than the specified length.  If this option is specified
      all non-pseudobond objects are also excluded.
    maximum_length : float or None
      Exclude pseudobonds longer than the specified length.  If this option is specified
      all non-pseudobond objects are also excluded.
    sequence : string or None
      Regular expression of sequence to match.  Will be automatically upcased.
    '''

    if objects is None:
        from chimerax.core.objects import all_objects
        objects = all_objects(session)

    from chimerax.core.undo import UndoState
    undo_state = UndoState("select")
    if sequence is None:
        objects = _filter_pseudobonds_by_length(objects, minimum_length, maximum_length)
        clear_selection(session, undo_state)
        modify_selection(objects, 'add', undo_state, full_residues = residues)
    else:
        clear_selection(session, undo_state)
        objects = _select_sequence(objects, sequence)
        modify_selection(objects, 'add', undo_state, full_residues = residues)

    session.undo.register(undo_state)
    report_selection(session)

    return objects

def _filter_pseudobonds_by_length(objects, minimum_length, maximum_length):
    if (minimum_length is None and maximum_length is None) or objects.num_pseudobonds == 0:
        return objects

    pbonds = objects.pseudobonds
    lengths = pbonds.lengths
    if minimum_length is not None and maximum_length is not None:
        from numpy import logical_and
        keep = logical_and((lengths >= minimum_length), (lengths <= maximum_length))
    elif minimum_length is not None:
        keep = (lengths >= minimum_length)
    elif maximum_length is not None:
        keep = (lengths <= maximum_length)
        
    from chimerax.core.objects import Objects
    fobj = Objects(pseudobonds = pbonds.filter(keep))
    return fobj

def _select_sequence(objects, sequence):
    from chimerax.atomic import Residues, Residue
    sel_residues = set()
    base_search_string = sequence.upper()
    protein_search_string = base_search_string.replace('B', '[DN]').replace('Z', '[EQ]')
    nucleic_search_string = base_search_string.replace('R', '[AG]').replace('Y', '[CTU]').replace(
        'N', '[ACGTU]')
    orig_res = objects.residues
    for chain in orig_res.chains.unique():
        search_string = protein_search_string \
            if chain.polymer_type == Residue.PT_PROTEIN else nucleic_search_string
        try:
            ranges = chain.search(search_string, case_sensitive=True)
        except ValueError as e:
            from chimerax.core.errors import UserError
            raise UserError(e)
        for start, length in ranges:
            sel_residues.update([r for r in chain.residues[start:start+length] if r])
    residues = Residues(sel_residues)
    atoms = residues.intersect(orig_res).atoms
    from chimerax.core.objects import Objects
    fobj = Objects(atoms = atoms, bonds = atoms.intra_bonds,
                   pseudobonds = atoms.intra_pseudobonds,
                   models = atoms.structures.unique())
    return fobj
    
def select_add(session, objects=None, residues=False):
    '''Add objects to the selection.
    If objects is None everything is selected.'''
    if objects is None:
        from chimerax.core.objects import all_objects
        objects = all_objects(session)
    from chimerax.core.undo import UndoState
    undo_state = UndoState("select add")
    modify_selection(objects, 'add', undo_state, full_residues = residues)
    session.undo.register(undo_state)
    report_selection(session)
    
def select_subtract(session, objects=None, residues=False):
    '''Subtract objects from the selection.
    If objects is None the selection is cleared.'''
    from chimerax.core.undo import UndoState
    undo_state = UndoState("select subtract")
    if objects is None:
        clear_selection(session, undo_state)
    else:
        modify_selection(objects, 'subtract', undo_state, full_residues = residues)
    session.undo.register(undo_state)
    report_selection(session)
    
def select_intersect(session, objects=None, residues=False):
    '''Reduce the selection by intersecting with specified objects.'''
    from chimerax.core.undo import UndoState
    undo_state = UndoState("select intersect")
    intersect_selection(objects, session, undo_state, full_residues = residues)
    session.undo.register(undo_state)
    report_selection(session)
    
def select_up(session):
    '''Extend the current selection up one level.'''
    session.selection.promote(session)
    report_selection(session)
    
def select_down(session):
    '''Reduce the current selection down one level. Only possible after extending selection.'''
    session.selection.demote(session)
    report_selection(session)
    
def select_clear(session):
    '''Clear the selection.'''
    from chimerax.core.undo import UndoState
    undo_state = UndoState("select clear")
    with session.triggers.block_trigger("selection changed"):
        clear_selection(session, undo_state)
    session.undo.register(undo_state)

def report_selection(session):
    # TODO: This routine is taking about 25% of time of select command with
    #       this example "open 6zm7 ; time sel nucleic".
    s = session.selection
    mlist = [m for m in s.models() if m.selected]	# Exclude grouping models
    mc = len(mlist)
    ac = bc = pbc = rc = 0
    for atoms in s.items('atoms'):
        ac += len(atoms)
        rc += atoms.num_residues
    for bonds in s.items('bonds'):
        bc += len(bonds)
    for pbonds in s.items('pseudobonds'):
        pbc += len(pbonds)
    lines = []
    if mc == 0 and ac == 0 and bc == 0 and pbc == 0:
        lines.append('Nothing')
    if ac != 0:
        plural = ('s' if ac > 1 else '')
        lines.append('%d atom%s' % (ac, plural))
    if bc != 0:
        plural = ('s' if bc > 1 else '')
        lines.append('%d bond%s' % (bc, plural))
    if pbc != 0:
        plural = ('s' if pbc > 1 else '')
        lines.append('%d pseudobond%s' % (pbc, plural))
    if rc != 0:
        plural = ('s' if rc > 1 else '')
        lines.append('%d residue%s' % (rc, plural))
    if mc != 0:
        plural = ('s' if mc > 1 else '')
        lines.append('%d model%s' % (mc, plural))
    session.logger.status(', '.join(lines) + ' selected', log = True)

def modify_selection(objects, mode, undo_state, full_residues = False):
    select = (mode == 'add')
    atoms, bonds, pbonds, models = _atoms_bonds_models(objects, full_residues = full_residues)
    if mode == 'subtract' and atoms:
        # don't leave partially selected bonds/pseudobonds
        from chimerax.atomic import concatenate, all_pseudobonds
        bonds = concatenate([bonds, atoms.bonds], remove_duplicates=True)
        session = atoms.structures[0].session
        all_pbs = all_pseudobonds(session)
        a1, a2 = all_pbs.atoms
        pbonds = concatenate([pbonds, all_pbs.filter(a1.mask(atoms) | a2.mask(atoms))],
            remove_duplicates=True)
    undo_state.add(atoms, "selected", atoms.selected, select)
    undo_state.add(bonds, "selected", bonds.selected, select)
    undo_state.add(pbonds, "selected", pbonds.selected, select)
    atoms.selected = select
    bonds.selected = select
    pbonds.selected = select
    for m in models:
        undo_state.add(m, "selected", m.selected, select)
        m.selected = select

def intersect_selection(objects, session, undo_state, full_residues = False):
    atoms, bonds, pbonds, models = _atoms_bonds_models(objects, full_residues = full_residues)
    from chimerax import atomic
    selatoms = atomic.selected_atoms(session)
    subatoms = selatoms - atoms
    selbonds = atomic.selected_bonds(session)
    subbonds = selbonds - bonds
    selpbonds = atomic.selected_pseudobonds(session)
    subpbonds = selpbonds - pbonds
    from chimerax.atomic import Structure, PseudobondGroup
    selmodels = set(m for m in session.selection.models()
                    if not isinstance(m, (Structure, PseudobondGroup)))
    submodels = selmodels.difference(models)
    undo_state.add(subatoms, "selected", subatoms.selected, False)
    undo_state.add(subbonds, "selected", subbonds.selected, False)
    undo_state.add(subpbonds, "selected", subpbonds.selected, False)
    subatoms.selected = False
    subbonds.selected = False
    subpbonds.selected = False
    for m in submodels:
        undo_state.add(m, "selected", m.selected, False)
        m.selected = False

def clear_selection(session, undo_state):
    session.selection.undo_add_selected(undo_state, False)
    session.selection.clear()

def _atoms_bonds_models(objects, full_residues = False):
    # Treat selecting molecular surface as selecting atoms.
    # Returned models list does not include atomic models
    atoms = objects.atoms
    bonds = objects.bonds
    pbonds = objects.pseudobonds
    satoms = []
    models = []
    from chimerax.atomic import MolecularSurface, Structure, PseudobondGroup
    for m in objects.models:
        if isinstance(m, MolecularSurface):
            if m.has_atom_patches():
                satoms.append(m.atoms)
            else:
                models.append(m)
        elif not isinstance(m, (Structure, PseudobondGroup)):
            models.append(m)
    if satoms:
        from chimerax.atomic import concatenate
        atoms = concatenate([atoms] + satoms)
    if full_residues:
        if len(bonds) > 0 or len(pbonds) > 0:
            a1, a2 = bonds.atoms
            pa1, pa2 = pbonds.atoms
            from chimerax.atomic import concatenate
            atoms = concatenate([atoms, a1, a2, pa1, pa2])
        atoms = atoms.unique_residues.atoms
        bonds = atoms.intra_bonds
    return atoms, bonds, pbonds, models

def register_command(logger):
    from chimerax.core.commands import CmdDesc, register, ObjectsArg, NoArg, create_alias, \
        BoolArg, FloatArg, StringArg
    from chimerax.atomic import AtomsArg
    desc = CmdDesc(optional=[('objects', ObjectsArg)],
                   keyword=[('residues', BoolArg),
                            ('minimum_length', FloatArg),
                            ('maximum_length', FloatArg),
                            ('sequence', StringArg)],
                   synopsis='select specified objects')
    register('select', desc, select, logger=logger)

    desc = CmdDesc(optional=[('objects', ObjectsArg)],
                   keyword=[('residues', BoolArg)],
                   synopsis='add objects to selection')
    register('select add', desc, select_add, logger=logger)

    desc = CmdDesc(optional=[('objects', ObjectsArg)],
                   keyword=[('residues', BoolArg)],
                   synopsis='subtract objects from selection')
    register('select subtract', desc, select_subtract, logger=logger)

    desc = CmdDesc(required=[('objects', ObjectsArg)],
                   keyword=[('residues', BoolArg)],
                   synopsis='intersect objects with selection')
    register('select intersect', desc, select_intersect, logger=logger)

    desc = CmdDesc(synopsis='extend the selection up one level')
    register('select up', desc, select_up, logger=logger)

    desc = CmdDesc(synopsis='revert the selection down one level')
    register('select down', desc, select_down, logger=logger)

    desc = CmdDesc(synopsis='clear the selection')
    register('select clear', desc, select_clear, logger=logger)

    create_alias('~select', 'select subtract $*', logger=logger)

    # Register "select zone" subcommand
    from . import zonesel
    zonesel.register_command(logger)
