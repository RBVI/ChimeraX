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

def select(session, objects=None, polymer=None, residues=False):
    '''Select specified objects.

    Parameters
    ----------
    objects : Objects
      Replace the current selection with the specified objects (typically atoms).
      If no objects are specified then everything is selected.
    residues : bool
      Extend atoms that are selected to containing residues if true (default false).
    polymer : Atoms
      Reduce the selection to include only atoms belonging to chains having a sequence that is the
      same as one of the sequences specified by the polymer option.
    '''

    if objects is None:
        from . import all_objects
        objects = all_objects(session)

    from ..undo import UndoState
    undo_state = UndoState("select")
    if objects is not None:
        clear_selection(session, "select_clear", undo_state)
        modify_selection(objects, 'add', undo_state, full_residues = residues)

    if polymer is not None:
        polymer_selection(polymer, session, undo_state)
        
    session.undo.register(undo_state)
    report_selection(session)

def select_add(session, objects=None, residues=False):
    '''Add objects to the selection.
    If objects is None everything is selected.'''
    if objects is None:
        from . import all_objects
        objects = all_objects(session)
    from ..undo import UndoState
    undo_state = UndoState("select add")
    modify_selection(objects, 'add', undo_state, full_residues = residues)
    session.undo.register(undo_state)
    report_selection(session)
    
def select_subtract(session, objects=None, residues=False):
    '''Subtract objects from the selection.
    If objects is None the selection is cleared.'''
    from ..undo import UndoState
    undo_state = UndoState("select subtract")
    if objects is None:
        clear_selection(session, "subtract_clear", undo_state)
    else:
        modify_selection(objects, 'subtract', undo_state, full_residues = residues)
    session.undo.register(undo_state)
    report_selection(session)
    
def select_intersect(session, objects=None, residues=False):
    '''Reduce the selection by intersecting with specified objects.'''
    from ..undo import UndoState
    undo_state = UndoState("select intersect")
    intersect_selection(objects, session, undo_state, full_residues = residues)
    session.undo.register(undo_state)
    report_selection(session)
    
def polymer_selection(seq_atoms, session, undo_state):
    '''
    Reduce the current selected atoms to include only those that belong to a chain
    having the same sequence string as one of seq_atoms.
    '''
    s = session.selection
    atoms_list = s.items('atoms')
    s.clear()
    if atoms_list:
        sseqs, sseq_ids = seq_atoms.residues.unique_sequences
        sset = set(sseqs)
        sset.discard('')	# Don't include non-polymers.
        for atoms in atoms_list:
            seqs, seq_ids = atoms.residues.unique_sequences
            from numpy import array, bool
            smask = array([(seq in sset) for seq in seqs], bool)
            satoms = atoms.filter(smask[seq_ids])
            undo_state.add(satoms, "selected", satoms.selected, True)
            satoms.selected = True
    
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
    clear_selection(session, "clear", undo_state)

def report_selection(session):
    s = session.selection
    from ..atomic import MolecularSurface, Structure    
    mc = len([m for m in s.models() if not isinstance(m, (Structure, MolecularSurface))])
    ac = sum([len(atoms) for atoms in s.items('atoms')], 0)
    bc = sum([len(bonds) for bonds in s.items('bonds')], 0)
    pbc = sum([len(pbonds) for pbonds in s.items('pseudobonds')], 0)
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
    if mc != 0:
        plural = ('s' if mc > 1 else '')
        lines.append('%d model%s' % (mc, plural))
    session.logger.status(', '.join(lines) + ' selected', log = True)

def modify_selection(objects, mode, undo_state, full_residues = False):
    select = (mode == 'add')
    atoms, bonds, pbonds, models = _atoms_bonds_models(objects, full_residues = full_residues)
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
    from .. import atomic
    selatoms = atomic.selected_atoms(session)
    subatoms = selatoms - atoms
    selbonds = atomic.selected_bonds(session)
    subbonds = selbonds - bonds
    selpbonds = atomic.selected_pseudobonds(session)
    subpbonds = selpbonds - pbonds
    from ..atomic import Structure
    selmodels = set(m for m in session.selection.models() if not isinstance(m, Structure))
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

def clear_selection(session, why, undo_state):
    from ..undo import UndoState
    undo_state = UndoState("select clear")
    session.selection.undo_add_selected(undo_state, False)
    session.selection.clear()
    session.undo.register(undo_state)

def _atoms_bonds_models(objects, full_residues = False):
    # Treat selecting molecular surface as selecting atoms.
    # Returned models list does not include atomic models
    atoms = objects.atoms
    bonds = objects.bonds
    pbonds = objects.pseudobonds
    satoms = []
    models = []
    from ..atomic import MolecularSurface, Structure, PseudobondGroup
    for m in objects.models:
        if isinstance(m, MolecularSurface):
            if m.has_atom_patches():
                satoms.append(m.atoms)
            else:
                models.append(m)
        elif not isinstance(m, (Structure, PseudobondGroup)):
            models.append(m)
    if satoms:
        from ..atomic import concatenate
        atoms = concatenate([atoms] + satoms)
    if full_residues:
        if len(bonds) > 0 or len(pbonds) > 0:
            a1, a2 = bonds.atoms
            pa1, pa2 = pbonds.atoms
            from ..atomic import concatenate
            atoms = concatenate([atoms, a1, a2, pa1, pa2])
        atoms = atoms.unique_residues.atoms
        bonds = atoms.intra_bonds
    return atoms, bonds, pbonds, models

def register_command(session):
    from . import CmdDesc, register, ObjectsArg, NoArg, create_alias, AtomsArg, BoolArg
    desc = CmdDesc(optional=[('objects', ObjectsArg)],
                   keyword=[('residues', BoolArg),
                            ('polymer', AtomsArg)],
                   synopsis='select specified objects')
    register('select', desc, select, logger=session.logger)

    desc = CmdDesc(optional=[('objects', ObjectsArg)],
                   keyword=[('residues', BoolArg)],
                   synopsis='add objects to selection')
    register('select add', desc, select_add, logger=session.logger)

    desc = CmdDesc(optional=[('objects', ObjectsArg)],
                   keyword=[('residues', BoolArg)],
                   synopsis='subtract objects from selection')
    register('select subtract', desc, select_subtract, logger=session.logger)

    desc = CmdDesc(required=[('objects', ObjectsArg)],
                   keyword=[('residues', BoolArg)],
                   synopsis='intersect objects with selection')
    register('select intersect', desc, select_intersect, logger=session.logger)

    desc = CmdDesc(synopsis='extend the selection up one level')
    register('select up', desc, select_up, logger=session.logger)

    desc = CmdDesc(synopsis='revert the selection down one level')
    register('select down', desc, select_down, logger=session.logger)

    desc = CmdDesc(synopsis='clear the selection')
    register('select clear', desc, select_clear, logger=session.logger)

    create_alias('~select', 'select subtract $*', logger=session.logger)

    # Register "select zone" subcommand
    from . import zonesel
    zonesel.register_command(session)
