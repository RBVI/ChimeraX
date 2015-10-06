# vi: set expandtab shiftwidth=4 softtabstop=4:


def register_core_commands(session):
    """Register core commands"""
    from importlib import import_module
    modules = ['buriedarea', 'camera', 'close', 'color', 'colordef', 'crossfade', 'crosslinks',
               'delete', 'display', 'echo', 'exit', 'export', 'lighting', 'list', 'material',
               'move', 'open', 'pdbimages', 'perframe', 'pwd', 'roll', 'run',
               'sasa', 'save', 'scolor', 'select', 'set', 'split', 'stop', 'style', 'surface', 'sym',
               'transparency', 'turn', 'usage', 'view', 'wait']
    for mod in modules:
        m = import_module('chimera.core.commands.%s' % mod)
        m.register_command(session)

    from .. import map
    map.register_volume_command()
    map.register_molmap_command()
    from ..map import filter
    filter.register_vop_command()
    from ..map import fit
    fit.register_fitmap_command()
    from ..map import series
    series.register_vseries_command()

    from ..devices import oculus
    oculus.register_oculus_command()
    from ..devices import spacenavigator
    spacenavigator.register_snav_command()

    # Selectors
    from . import atomspec
    atomspec.register_selector(None, "sel", _sel_selector)
    atomspec.register_selector(None, "strands", _strands_selector)
    atomspec.register_selector(None, "ions",
        lambda s, m, r: _structure_category_selector("ions", m, r))
    atomspec.register_selector(None, "ligand",
        lambda s, m, r: _structure_category_selector("ligand", m, r))
    atomspec.register_selector(None, "main",
        lambda s, m, r: _structure_category_selector("main", m, r))
    atomspec.register_selector(None, "solvent",
        lambda s, m, r: _structure_category_selector("solvent", m, r))
    from ..atomic.molobject import Element
    for i in range(1, 115):
        e = Element.get_element(i)
        atomspec.register_selector(None, e.name,
            lambda ses, models, results, sym=e.name: _element_selector(sym, models, results))


def _element_selector(symbol, models, results):
    from ..atomic import AtomicStructure
    for m in models:
        if isinstance(m, AtomicStructure):
            atoms = m.atoms.filter(m.atoms.element_names == symbol)
            if len(atoms) > 0:
                results.add_model(m)
                results.add_atoms(atoms)

def _sel_selector(session, models, results):
    from ..atomic import AtomicStructure
    for m in models:
        if m.any_part_selected():
            results.add_model(m)
            if isinstance(m, AtomicStructure):
                for atoms in m.selected_items('atoms'):
                    results.add_atoms(atoms)


def _strands_selector(session, models, results):
    from ..atomic import AtomicStructure
    for m in models:
        if isinstance(m, AtomicStructure):
            strands = m.residues.filter(m.residues.is_sheet)
            if strands:
                results.add_model(m)
                results.add_atoms(strands.atoms)

def _structure_category_selector(cat, models, results):
    from ..atomic import AtomicStructure
    for m in models:
        if isinstance(m, AtomicStructure):
            atoms = m.atoms.filter(m.atoms.structure_categories == cat)
            if len(atoms) > 0:
                results.add_model(m)
                results.add_atoms(atoms)

