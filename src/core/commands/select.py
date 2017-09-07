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

def select(session, objects=None, polymer=None):
    '''Select specified objects.

    Parameters
    ----------
    objects : Objects
      Replace the current selection with the specified objects (typically atoms).
      If no objects are specified then everything is selected.
    polymer : Atoms
      Reduce the selection to include only atoms belonging to chains having a sequence that is the
      same as one of the sequences specified by the polymer option.
    '''

    if objects is None:
        from . import all_objects
        objects = all_objects(session)

    undo_data = _undo_setup()
    if objects is not None:
        clear_selection(session, "select_clear", undo_data)
        modify_selection(objects, 'add', undo_data)

    if polymer is not None:
        polymer_selection(polymer, session, undo_data)
    _undo_finish(session, undo_data)
        
    report_selection(session)

def select_add(session, objects=None):
    '''Add objects to the selection.
    If objects is None everything is selected.'''
    if objects is None:
        from . import all_objects
        objects = all_objects(session)
    undo_data = _undo_setup()
    modify_selection(objects, 'add', undo_data)
    _undo_finish(session, undo_data)

def select_subtract(session, objects=None):
    '''Subtract objects from the selection.
    If objects is None the selection is cleared.'''
    undo_data = _undo_setup()
    if objects is None:
        clear_selection(session, "subtract_clear", undo_data)
    else:
        modify_selection(objects, 'subtract', undo_data)
    _undo_finish(session, undo_data)

def select_intersect(session, objects=None):
    '''Reduce the selection by intersecting with specified objects.'''
    undo_data = _undo_setup()
    intersect_selection(objects, session, undo_data)
    _undo_finish(session, undo_data)

def polymer_selection(seq_atoms, session, undo_data):
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
        undo_data["atoms"]["polymer"] = up = []
        for atoms in atoms_list:
            seqs, seq_ids = atoms.residues.unique_sequences
            from numpy import array, bool
            smask = array([(seq in sset) for seq in seqs], bool)
            satoms = atoms.filter(smask[seq_ids])
            up.append((satoms, satoms.selected, True))
            satoms.selected = True
    
def select_clear(session, objects=None):
    '''Clear the selection.'''
    undo_data = _undo_setup()
    clear_selection(session, "clear", undo_data)
    _undo_finish(session, undo_data)

def report_selection(session):
    s = session.selection
    from ..atomic import MolecularSurface, Structure    
    mc = len([m for m in s.models() if not isinstance(m, (Structure, MolecularSurface))])
    ac = sum([len(atoms) for atoms in s.items('atoms')], 0)
    lines = []
    if mc == 0 and ac == 0:
        lines.append('Nothing')
    if ac != 0:
        plural = ('s' if ac > 1 else '')
        lines.append('%d atom%s' % (ac, plural))
    if mc != 0:
        plural = ('s' if mc > 1 else '')
        lines.append('%d model%s' % (mc, plural))
    session.logger.status(', '.join(lines) + ' selected', log = True)

def modify_selection(objects, mode, undo_data):
    select = (mode == 'add')
    atoms, models = _atoms_and_models(objects)
    undo_data["atoms"]["modify"] = [(atoms, atoms.selected, select)]
    atoms.selected = select
    undo_data["models"]["modify"] = um = []
    for m in models:
        um.append((m, m.selected, select))
        m.selected = select

def intersect_selection(objects, session, undo_data):
    atoms, models = _atoms_and_models(objects)
    from .. import atomic
    selatoms = atomic.selected_atoms(session)
    subatoms = selatoms - atoms
    from ..atomic import Structure
    selmodels = set(m for m in session.selection.models() if not isinstance(m, Structure))
    submodels = selmodels.difference(models)
    undo_data["atoms"]["intersect"] = [(subatoms, subatoms.selected, False)]
    subatoms.selected = False
    undo_data["models"]["intersect"] = um = []
    for m in submodels:
        um.append((m, m.selected, False))
        m.selected = False

def clear_selection(session, why, undo_data):
    from ..atomic.molarray import Atoms
    atoms = session.selection.items("atoms")
    if atoms:
        undo_data["atoms"][why] = ua = []
        if isinstance(atoms, Atoms):
            ua.append((atoms, atoms.selected, False))
        else:
            for a in atoms:
                ua.append((a, a.selected, False))
    models = [m for m in session.selection.all_models() if m.selected]
    if models:
        undo_data["models"][why] = um = []
        for m in models:
            um.append((m, m.selected, False))
    session.selection.clear()

def _atoms_and_models(objects):
    # Treat selecting molecular surface as selecting atoms.
    # Returned models list does not include atomic models
    atoms = objects.atoms
    satoms = []
    models = []
    from ..atomic import MolecularSurface, Structure
    for m in objects.models:
        if isinstance(m, MolecularSurface):
            if m.has_atom_patches():
                satoms.append(m.atoms)
            else:
                models.append(m)
        elif not isinstance(m, Structure):
            models.append(m)
    if satoms:
        from ..atomic import molarray
        atoms = molarray.concatenate([atoms] + satoms)
    return atoms, models

def _undo_setup():
    undo_data = {}
    undo_data["models"] = {}
    undo_data["atoms"] = {}
    return undo_data

def _undo_finish(session, undo_data):
    def undo(undo_data=undo_data):
        _selection_undo(undo_data)
    def redo(undo_data=undo_data):
        _selection_redo(undo_data)
    session.undo.register("selection", undo, redo)

def _update_selected(undo_dict, key, attr, n):
    try:
        items = undo_dict[key]
    except KeyError:
        return
    for item in items:
        container = item[0]
        value = item[n]
        setattr(container, attr, value)

def _selection_undo(undo_data):
    atoms_dict = undo_data["atoms"]
    models_dict = undo_data["models"]

    _update_selected(models_dict, "modify", "selected", 1)
    _update_selected(models_dict, "intersect", "selected", 1)
    _update_selected(atoms_dict, "polymer", "selected", 1)
    _update_selected(atoms_dict, "modify", "selected", 1)
    _update_selected(atoms_dict, "intersect", "selected", 1)

    _update_selected(models_dict, "select_clear", "selected", 1)
    _update_selected(models_dict, "subtract_clear", "selected", 1)
    _update_selected(models_dict, "clear", "selected", 1)
    _update_selected(atoms_dict, "select_clear", "selected", 1)
    _update_selected(atoms_dict, "subtract_clear", "selected", 1)
    _update_selected(atoms_dict, "clear", "selected", 1)

def _selection_redo(undo_data):
    atoms_dict = undo_data["atoms"]
    models_dict = undo_data["models"]

    _update_selected(models_dict, "select_clear", "selected", 2)
    _update_selected(models_dict, "subtract_clear", "selected", 2)
    _update_selected(models_dict, "clear", "selected", 2)
    _update_selected(atoms_dict, "select_clear", "selected", 2)
    _update_selected(atoms_dict, "subtract_clear", "selected", 2)
    _update_selected(atoms_dict, "clear", "selected", 2)

    _update_selected(models_dict, "modify", "selected", 2)
    _update_selected(models_dict, "intersect", "selected", 2)
    _update_selected(atoms_dict, "polymer", "selected", 2)
    _update_selected(atoms_dict, "modify", "selected", 2)
    _update_selected(atoms_dict, "intersect", "selected", 2)

def register_command(session):
    from . import CmdDesc, register, ObjectsArg, NoArg, create_alias, AtomsArg
    desc = CmdDesc(optional=[('objects', ObjectsArg)],
                   keyword=[('polymer', AtomsArg)],
                   synopsis='select specified objects')
    register('select', desc, select, logger=session.logger)

    desc = CmdDesc(optional=[('objects', ObjectsArg)],
                   synopsis='add objects to selection')
    register('select add', desc, select_add, logger=session.logger)

    desc = CmdDesc(optional=[('objects', ObjectsArg)],
                   synopsis='subtract objects from selection')
    register('select subtract', desc, select_subtract, logger=session.logger)

    desc = CmdDesc(required=[('objects', ObjectsArg)],
                   synopsis='intersect objects with selection')
    register('select intersect', desc, select_intersect, logger=session.logger)

    desc = CmdDesc(synopsis='clear the selection')
    register('select clear', desc, select_clear, logger=session.logger)

    create_alias('~select', 'select subtract $*', logger=session.logger)

    # Register "select zone" subcommand
    from . import zonesel
    zonesel.register_command(session)
