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

_SpecialColors = ["byatom", "byelement", "byhetero", "bychain",
                  "bypolymer", "byidentity", "bynucleotide", "bymodel",
                  "fromatoms", "fromcartoons", "fromribbons", "random"]

_SequentialLevels = ["residues", "chains", "polymers", "structures"]
# More possible sequential levels: "helix", "helices", "strands", "SSEs", "volmodels", "allmodels"

DEFAULT_TARGETS = 'acsbpf'
ALL_TARGETS = 'acrsbmpfl'
WHAT_TARGETS = {
    'atoms': 'a',
    'cartoons': 'c', 'ribbons': 'c',
    'surfaces': 's',
    'bonds': 'b',
    'models': 'm',
    'pseudobonds': 'p',
    'rings': 'f',
    'labels': 'l',
    'All': ALL_TARGETS
}


def get_targets(targets, what, default_targets=DEFAULT_TARGETS):
    if targets is None and what is None:
        return default_targets, True
    if ((targets is not None and 'A' in targets) or
            (what is not None and 'All' in what)):
        return ALL_TARGETS, True
    if targets and 'r' in targets:
        targets += 'c'
    if what is not None:
        if targets is None:
            targets = ''
        for w in what:
            targets += WHAT_TARGETS[w]
    return targets, False


def color(session, objects, color=None, what=None, target=None,
          transparency=None, halfbond=None, undo_name="color"):
    """Color atoms, ribbons, surfaces, ....

    Parameters
    ----------
    objects : Objects
      Which objects to color.
    color : Color
      Color can be a standard color name or "byatom", "byelement", "byhetero", "bychain", "bypolymer", "byidentity", "bynucleotide", "bymodel".
    what :  'atoms', 'cartoons', 'ribbons', 'surfaces', 'bonds', 'pseudobonds', 'labels', 'models' or None
      What to color. Everything is colored if option is not specified.
    target : string containing letters 'a', 'b', 'c', 'p', 'r', 's', 'f', 'm'
      Alternative to the "what" option for specifying what to color.
      Characters indicating what to color, a = atoms, c = cartoon, r = cartoon, s = surfaces,
      b = bonds, p = pseudobonds, f = (filled) rings, m = models
      Everything except labels and models is colored if no target is specified.
    transparency : float
      Percent transparency to use.  If not specified current transparency is preserved.
    halfbond : bool
      Whether to color each half of a bond to match the connected atoms.
      If halfbond is false the bond is given the single color assigned to the bond.
    """
    if objects is None:
        from chimerax.core.objects import all_objects
        objects = all_objects(session)

    default_targets = 'l' if _only_2d_labels(objects) else DEFAULT_TARGETS 
    target, is_default_target = get_targets(target, what, default_targets = default_targets)

    from chimerax.core.undo import UndoState
    undo_state = UndoState(undo_name)

    # Decide whether to set or preserve transparency
    opacity = None
    if transparency is not None:
        opacity = min(255, max(0, int(2.56 * (100 - transparency))))
    elif getattr(color, 'explicit_transparency', False):
        opacity = color.uint8x4()[3]

    if halfbond is not None:
        bonds = objects.bonds
        if len(bonds) > 0:
            undo_state.add(bonds, "halfbonds", bonds.halfbonds, halfbond)
            bonds.halfbonds = halfbond
        if 'p' in target:
            pbonds = objects.pseudobonds
            if len(pbonds) > 0:
                undo_state.add(pbonds, "halfbonds", pbonds.halfbonds, halfbond)
                pbonds.halfbonds = halfbond

    items = []

    bgcolor = session.main_view.background_color

    if 'a' in target or 's' in target:
        atoms = objects.atoms
        if color == "byhetero":
            atoms = atoms.filter(atoms.element_numbers != 6)

    if 'a' in target:
        # atoms/bonds
        if atoms is not None and color is not None:
            if _set_atom_colors(atoms, color, opacity, bgcolor, undo_state):
                items.append('%d atoms' % len(atoms))

    if 's' in target and color is not None:
        from chimerax.atomic import MolecularSurface, concatenate, Structure, PseudobondGroup
        msatoms = [m.atoms for m in objects.models
                   if isinstance(m, MolecularSurface) and not m.atoms.intersects(atoms)]
        satoms = concatenate(msatoms + [atoms]) if msatoms else atoms
        if color == "byhetero":
            satoms = satoms.filter(satoms.element_numbers != 6)
        if color == "bynucleotide":
            from chimerax.atomic import Residue
            satoms = satoms.filter(satoms.residues.polymer_types == Residue.PT_NUCLEIC)
        ns = _set_surface_colors(session, satoms, color, opacity, bgcolor,
                                 undo_state=undo_state)
        # Handle non-molecular surfaces like density maps
        if color not in _SpecialColors:
            from chimerax.core.models import Surface
            surfs = [m for m in objects.models
                     if isinstance(m, Surface) and not isinstance(m, MolecularSurface)]
            _set_model_colors(session, surfs, color, opacity, undo_state)
            ns += len(surfs)
        items.append('%d surfaces' % ns)

    residues = None
    if 'c' in target and color is not None:
        residues = objects.residues
        if _set_ribbon_colors(residues, color, opacity, bgcolor, undo_state):
            items.append('%d residues' % len(residues))

    if 'f' in target and color is not None:
        if residues is None:
            residues = objects.residues
        if _set_ring_colors(residues, color, opacity, bgcolor, undo_state):
            items.append('rings')  # not sure how many

    if 'b' in target:
        if color not in _SpecialColors and color is not None:
            bonds = objects.bonds
            if len(bonds) > 0:
                if color not in _SpecialColors:
                    color_array = color.uint8x4()
                    undo_state.add(bonds, "colors", bonds.colors, color_array)
                    bonds.colors = color_array
                    items.append('%d bonds' % len(bonds))

    if 'p' in target:
        if color not in _SpecialColors and color is not None:
            pbonds = objects.pseudobonds
            if len(pbonds) > 0:
                color_array = color.uint8x4()
                undo_state.add(pbonds, "colors", pbonds.colors, color_array)
                pbonds.colors = color_array
                items.append('%d pseudobonds' % len(pbonds))

    if 'l' in target:
        if color not in _SpecialColors:
            nl = _set_label_colors(session, objects, color, opacity, undo_state=undo_state)
            if nl > 0:
                items.append('%d labels' % nl)
    
    if 'm' in target:
        if color not in _SpecialColors:
            _set_model_colors(session, objects.models, color, opacity, undo_state)
            items.append('%d models' % len(objects.models))

    if not items:
        items.append('nothing')

    from chimerax.core.commands import commas
    session.logger.status('Colored %s' % commas(items, 'and'))
    session.undo.register(undo_state)


def _computed_atom_colors(atoms, color, opacity, bgcolor):
    if color in ("byatom", "byelement", "byhetero"):
        c = _element_colors(atoms, opacity)
    elif color == "bychain":
        from chimerax.atomic.colors import chain_colors
        c = chain_colors(atoms.residues.chain_ids)
        c[:, 3] = atoms.colors[:, 3] if opacity is None else opacity
    elif color == "bypolymer" or color == "byidentity":
        from chimerax.atomic.colors import polymer_colors
        c = atoms.colors.copy()
        sc,amask = polymer_colors(atoms.residues)
        c[amask,:] = sc[amask,:]
        c[amask, 3] = atoms.colors[amask, 3] if opacity is None else opacity
    elif color == "bynucleotide":
        from chimerax.atomic.colors import nucleotide_colors
        c = atoms.colors.copy()
        sc, amask = nucleotide_colors(atoms.residues)
        c[amask, :] = sc[amask, :]
        c[amask, 3] = atoms.colors[amask, 3] if opacity is None else opacity
    elif color == "bymodel":
        c = atoms.colors.copy()
        for m, matoms in atoms.by_structure:
            color = m.initial_color(bgcolor).uint8x4()
            mi = atoms.mask(matoms)
            c[mi, :3] = color[:3]
            if opacity is not None:
                c[mi, 3] = opacity
    elif color == "random":
        from numpy import random, uint8
        c = random.randint(0, 255, (len(atoms), 4)).astype(uint8)
        c[:, 3] = 255   # Opaque
    else:
        # Other "colors" do not apply to atoms
        c = None
    return c

def _only_2d_labels(objects):
    have_2d_label = False
    from chimerax.label.label2d import Labels, LabelModel
    for m in objects.models:
        if isinstance(m, (Labels, LabelModel)):
            have_2d_label = True
        else:
            return False
    if not have_2d_label:
        return False
        
    if (objects.num_atoms > 0 or
        objects.num_bonds > 0 or
        objects.num_pseudobonds > 0):
        return False
    return True

def _element_colors(atoms, opacity=None):
    from chimerax.atomic.colors import element_colors
    c = element_colors(atoms.element_numbers)
    c[:, 3] = atoms.colors[:, 3] if opacity is None else opacity
    return c


def _set_atom_colors(atoms, color, opacity, bgcolor, undo_state):
    if color in _SpecialColors:
        c = _computed_atom_colors(atoms, color, opacity, bgcolor)
        if c is not None:
            undo_state.add(atoms, "colors", atoms.colors, c)
            atoms.colors = c
            return True
    else:
        c = atoms.colors
        c[:, :3] = color.uint8x4()[:3]    # Preserve transparency
        if opacity is not None:
            c[:, 3] = opacity
        undo_state.add(atoms, "colors", atoms.colors, c)
        atoms.colors = c
        return True
    return False

def _set_ribbon_colors(residues, color, opacity, bgcolor, undo_state):
    if color not in _SpecialColors:
        c = residues.ribbon_colors
        c[:, :3] = color.uint8x4()[:3]    # Preserve transparency
        if opacity is not None:
            c[:, 3] = opacity
        undo_state.add(residues, "ribbon_colors", residues.ribbon_colors, c)
        residues.ribbon_colors = c
    elif color == 'bychain':
        from chimerax.atomic.colors import chain_colors
        c = chain_colors(residues.chain_ids)
        c[:, 3] = residues.ribbon_colors[:, 3] if opacity is None else opacity
        undo_state.add(residues, "ribbon_colors", residues.ribbon_colors, c)
        residues.ribbon_colors = c
    elif color == "bypolymer" or color == "byidentity":
        from chimerax.atomic.colors import polymer_colors
        c,rmask = polymer_colors(residues)
        c[rmask, 3] = residues.ribbon_colors[rmask, 3] if opacity is None else opacity
        masked_residues = residues.filter(rmask)
        undo_state.add(masked_residues, "ribbon_colors", masked_residues.ribbon_colors, c[rmask,:])
        masked_residues.ribbon_colors = c[rmask,:]
    elif color == "bynucleotide":
        from chimerax.atomic.colors import nucleotide_colors
        c,rmask = nucleotide_colors(residues)
        c[rmask, 3] = residues.ribbon_colors[rmask, 3] if opacity is None else opacity
        masked_residues = residues.filter(rmask)
        undo_state.add(masked_residues, "ribbon_colors", masked_residues.ribbon_colors, c[rmask, :])
        masked_residues.ribbon_colors = c[rmask, :]
    elif color == 'bymodel':
        for m, res in residues.by_structure:
            c = res.ribbon_colors
            c[:, :3] = m.initial_color(bgcolor).uint8x4()[:3]
            if opacity is not None:
                c[:, 3] = opacity
            undo_state.add(res, "ribbon_colors", res.ribbon_colors, c)
            res.ribbon_colors = c
    elif color == 'random':
        from numpy import random, uint8
        c = random.randint(0, 255, (len(residues), 4)).astype(uint8)
        c[:, 3] = 255   # No transparency
        undo_state.add(residues, "ribbon_colors", residues.ribbon_colors, c)
        residues.ribbon_colors = c
    else:
        return False
    return True

def _set_ring_colors(residues, color, opacity, bgcolor, undo_state):
    if color not in _SpecialColors:
        c = residues.ring_colors
        c[:, :3] = color.uint8x4()[:3]    # Preserve transparency
        if opacity is not None:
            c[:, 3] = opacity
        undo_state.add(residues, "ring_colors", residues.ring_colors, c)
        residues.ring_colors = c
    elif color == 'bychain':
        from chimerax.atomic.colors import chain_colors
        c = chain_colors(residues.chain_ids)
        c[:, 3] = residues.ring_colors[:, 3] if opacity is None else opacity
        undo_state.add(residues, "ring_colors", residues.ring_colors, c)
        residues.ring_colors = c
    elif color == "bypolymer" or color == "byidentity":
        from chimerax.atomic.colors import polymer_colors
        c,rmask = polymer_colors(residues)
        c[rmask, 3] = residues.ring_colors[rmask, 3] if opacity is None else opacity
        masked_residues = residues.filter(rmask)
        undo_state.add(masked_residues, "ring_colors", masked_residues.ring_colors, c[rmask,:])
        masked_residues.ring_colors = c[rmask,:]
    elif color == "bynucleotide":
        from chimerax.atomic.colors import nucleotide_colors
        c,rmask = nucleotide_colors(residues)
        c[rmask, 3] = residues.ring_colors[rmask, 3] if opacity is None else opacity
        masked_residues = residues.filter(rmask)
        undo_state.add(masked_residues, "ring_colors", masked_residues.ring_colors, c[rmask, :])
        masked_residues.ring_colors = c[rmask, :]
    elif color == 'bymodel':
        for m, res in residues.by_structure:
            c = res.ring_colors
            c[:, :3] = m.initial_color(bgcolor).uint8x4()[:3]
            if opacity is not None:
                c[:, 3] = opacity
            undo_state.add(res, "ring_colors", res.ring_colors, c)
            res.ring_colors = c
    elif color == 'random':
        from numpy import random, uint8
        c = random.randint(0, 255, (len(residues), 4)).astype(uint8)
        c[:, 3] = 255   # No transparency
        undo_state.add(residues, "ring_colors", residues.ring_colors, c)
        residues.ring_colors = c
    else:
        return False
    return True

def _set_surface_colors(session, atoms, color, opacity, bgcolor=None, undo_state=None):
    if color in _SpecialColors:
        if color == 'fromatoms':
            ns = _color_surfaces_at_atoms(atoms, opacity=opacity, undo_state=undo_state)
        elif color == 'fromribbons' or color == "fromcartoons":
            res = atoms.unique_residues
            ns = _color_surfaces_at_residues(res, res.ribbon_colors, opacity=opacity,
                                             undo_state=undo_state)
        else:
            # Surface colored different from atoms
            c = _computed_atom_colors(atoms, color, opacity, bgcolor)
            ns = _color_surfaces_at_atoms(atoms, per_atom_colors=c, opacity=opacity,
                                          undo_state=undo_state)
    else:
        ns = _color_surfaces_at_atoms(atoms, color.uint8x4(), opacity=opacity,
                                      undo_state=undo_state)
    return ns

def _set_model_colors(session, model_list, color, opacity, undo_state):
    for m in model_list:
        if undo_state:
            cprev = m.color_undo_state
        c = color.uint8x4()
        if not opacity is None:
            c[3] = opacity
        elif not m.model_color is None and not m.model_color is False:
            c[3] = m.model_color[3]
        m.model_color = c
        if undo_state:
            undo_state.add(m, 'color_undo_state', cprev, m.color_undo_state)

def _set_label_colors(session, objects, color, opacity, undo_state=None):
    nl = 0

    # 2D labels
    from chimerax.label.label2d import LabelModel
    labels = [m for m in objects.models if isinstance(m, LabelModel)]
    if undo_state:
        old_colors = [label.model_color for label in labels]
    for label in labels:
        label.model_color = _color_with_opacity(color, opacity, label.color)
    if undo_state:
        new_colors = [label.model_color for label in labels]
        for label, old_color, new_color in zip(labels, old_colors, new_colors):
            undo_state.add(label, 'model_color', old_color, new_color)
    nl += len(labels)

    # 3D labels
    from chimerax.label.label3d import label_objects
    lmodels, lobjects = label_objects(objects)
    rgba = color.uint8x4()
    if undo_state:
        for lo in lobjects:
            undo_state.add(lo, 'color', lo.color, rgba)
        for lm in lmodels:
            undo_state.add(lm, 'update_labels', (), (), option = 'MA')
    for lo in lobjects:
        lo.color = rgba
    for lm in lmodels:
        lm.update_labels()
    nl += len(lobjects)

    return nl

def _color_with_opacity(color, opacity, rgba):
    c = color.uint8x4()
    c[3] = rgba[3] if opacity is None else opacity
    return c
    
# -----------------------------------------------------------------------------
# Chain ids in each structure are colored from color map ordered alphabetically.
#
def _set_sequential_chain(session, objects, cmap, opacity, target, undo_state):
    # Organize atoms by structure and then chain
    uc = objects.residues.chains.unique()
    chain_atoms = {}
    for c in uc:
        chain_atoms.setdefault(c.structure, []).append((c.chain_id, c.existing_residues.atoms))

    # Make sure there is a colormap
    if cmap is None:
        from chimerax.core import colors
        cmap = colors.BuiltinColormaps["rainbow"]
    use_color_alpha = (opacity is None) and cmap.is_transparent

    # Each structure is colored separately with cmap applied by chain
    import numpy
    from chimerax.core.colors import Color
    for sl in chain_atoms.values():
        sl.sort(key = lambda ca: ca[0])	# Sort by chain id
        colors = cmap.interpolated_rgba(numpy.linspace(0.0, 1.0, len(sl)))
        for color, (chain_id, atoms) in zip(colors, sl):
            c = Color(color)
            opac = 255*color[3] if use_color_alpha else opacity
            if target is None or 'a' in target:
                _set_atom_colors(atoms, c, opac, None, undo_state)
            if target is None or 'c' in target:
                res = atoms.unique_residues
                _set_ribbon_colors(res, c, opac, None, undo_state)
            if target is None or 'f' in target:
                res = atoms.unique_residues
                _set_ring_colors(res, c, opac, None, undo_state)
            if target is None or 's' in target:
                _set_surface_colors(session, atoms, c, opac, undo_state=undo_state)

# ----------------------------------------------------------------------------------
# Polymers (unique sequences) in each structure are colored from color map ordered
# by polymer length.
#
def _set_sequential_polymer(session, objects, cmap, opacity, target, undo_state):
    # Organize atoms by structure and then polymer sequence
    uc = objects.residues.chains.unique()
    seq_atoms = {}
    for c in uc:
        seq_atoms.setdefault(c.structure, {}).setdefault(c.characters, []).append(c.existing_residues.atoms)

    # Make sure there is a colormap
    if cmap is None:
        from chimerax.core import colors
        cmap = colors.BuiltinColormaps["rainbow"]
    use_color_alpha = (opacity is None) and cmap.is_transparent

    # Each structure is colored separately with cmap applied by chain
    import numpy
    from chimerax.core.colors import Color
    for sl in seq_atoms.values():
        sseq = list(sl.items())
        sseq.sort(key = lambda sa: len(sa[0]))	# Sort by sequence length
        colors = cmap.interpolated_rgba(numpy.linspace(0.0, 1.0, len(sseq)))
        for color, (seq, alist) in zip(colors, sseq):
            c = Color(color)
            opac = 255*color[3] if use_color_alpha else opacity
            for atoms in alist:
                if target is None or 'a' in target:
                    _set_atom_colors(atoms, c, opac, None, undo_state)
                if target is None or 'c' in target:
                    res = atoms.unique_residues
                    _set_ribbon_colors(res, c, opac, None, undo_state)
                if target is None or 'f' in target:
                    res = atoms.unique_residues
                    _set_ring_colors(res, c, opac, None, undo_state)
                if target is None or 's' in target:
                    _set_surface_colors(session, atoms, c, opac, undo_state=undo_state)

# -----------------------------------------------------------------------------
#
def _set_sequential_residue(session, objects, cmap, opacity, target, undo_state):
    # Make sure there is a colormap
    if cmap is None:
        from chimerax.core import colors
        cmap = colors.BuiltinColormaps["rainbow"]
    use_color_alpha = (opacity is None) and cmap.is_transparent

    # Get chains and atoms in chains with "by_chain"
    # Each chain is colored separately with cmap applied by residue
    res = objects.atoms.unique_residues
    chain_res = [(chain, chain.existing_residues.intersect(res))
                 for chain in res.unique_chains]
    import numpy
    from chimerax.core.colors import Color
    for chain, residues in chain_res:
        colors = cmap.interpolated_rgba8(numpy.linspace(0.0, 1.0, len(residues)))
        for color, r in zip(colors, residues):
            c = Color(color)
            opac = color[3] if use_color_alpha else opacity
            if target is None or 'a' in target:
                _set_atom_colors(r.atoms, c, opac, None, undo_state)
            if target is None or 'c' in target:
                rgba = c.uint8x4()
                rgba[3] = r.ribbon_color[3] if opac is None else opac
                undo_state.add(r, "ribbon_color", r.ribbon_color, rgba)
                r.ribbon_color = rgba
        if 's' in target:
            _color_surfaces_at_residues(residues, colors, opacity=opac,
                                        undo_state = undo_state)
                
# -----------------------------------------------------------------------------
#
def _set_sequential_structures(session, objects, cmap, opacity, target, undo_state):
    # Make sure there is a colormap
    if cmap is None:
        from chimerax.core import colors
        cmap = colors.BuiltinColormaps["rainbow"]
    use_color_alpha = (opacity is None) and cmap.is_transparent

    from chimerax.atomic import Structure
    models = list(m for m in objects.models if isinstance(m, Structure))
    models.sort(key = lambda m: m.id)
    if len(models) == 0:
        return

    # Each structure is colored separately with cmap applied by chain
    import numpy
    from chimerax.core.colors import Color, rgba_to_rgba8
    colors = cmap.interpolated_rgba(numpy.linspace(0.0, 1.0, len(models)))
    for color, m in zip(colors, models):
        c = Color(color)
        opac = 255*color[3] if use_color_alpha else opacity
        if 'a' in target:
            _set_atom_colors(m.atoms, c, opac, None, undo_state)
        if 'c' in target:
            _set_ribbon_colors(m.residues, c, opac, None, undo_state)
        if 'f' in target:
            _set_ring_colors(m.residues, c, opac, None, undo_state)
        if 's' in target:
            _color_surfaces_at_atoms(m.atoms, rgba_to_rgba8(color),
                                     opacity=opac, undo_state=undo_state)

# -----------------------------------------------------------------------------
#
_SequentialColor = {
    "polymers": _set_sequential_polymer,
    "chains": _set_sequential_chain,
    "residues": _set_sequential_residue,
    "structures": _set_sequential_structures,
}

# -----------------------------------------------------------------------------
#
def color_func(session, objects, what=None, target=None, func=None, func_text='Changed ', undo_name='color modify'):
    if objects is None:
        from chimerax.core.objects import all_objects
        objects = all_objects(session)

    from chimerax.core.undo import UndoState
    undo_state = UndoState(undo_name)

    target, is_default_target = get_targets(target, what)

    what = []

    if 'a' in target or 's' in target:
        atoms = objects.atoms

    if 'a' in target:
        # atoms/bonds
        c = func(atoms.colors)
        undo_state.add(atoms, "colors", atoms.colors, c)
        atoms.colors = c
        what.append('%d atoms' % len(atoms))

    if 's' in target:
        surfs = _set_surface_color_func(atoms, objects, session, func, undo_state)
        what.append('%d surfaces' % len(surfs))

    if 'c' in target:
        residues = objects.residues
        c = func(residues.ribbon_colors)
        undo_state.add(residues, "ribbon_colors", residues.ribbon_colors, c)
        residues.ribbon_colors = c
        what.append('%d residues' % len(residues))

    if 'f' in target:
        residues = objects.residues
        c = func(residues.ring_colors)
        undo_state.add(residues, "ring_colors", residues.ring_colors, c)
        residues.ring_colors = c
        # TODO: what.append('%d residues' % len(residues))

    if 'b' in target:
        bonds = objects.bonds
        if len(bonds) > 0:
            c = func(bonds.colors)
            undo_state.add(bonds, "colors", bonds.colors, c)
            bonds.colors = c
            what.append('%d bonds' % len(bonds))


    if 'p' in target:
        pbonds = objects.pseudobonds
        if len(pbonds) > 0:
            c = func(pbonds.colors)
            undo_state.add(pbonds, "colors", pbonds.colors, c)
            pbonds.colors = c
            what.append('%d pseudobonds' % len(pbonds))

    if not what:
        what.append('nothing')

    from chimerax.core.commands import commas
    session.logger.status('%s %s' % (func_text, commas(what, 'and')))
    session.undo.register(undo_state)

def _set_surface_color_func(atoms, objects, session, func, undo_state=None):
    # TODO: save undo data

    # Handle surfaces for specified atoms
    from chimerax import atomic
    surfs = atomic.surfaces_with_atoms(atoms)
    for s in surfs:
        vcolors = s.vertex_colors
        amask = s.atoms.mask(atoms)
        all_atoms = amask.all()
        if all_atoms:
            c = func(s.colors)
            s.colors = c
            if vcolors is None:
                continue

        if vcolors is None:
            from numpy import empty, uint8
            vcolors = empty((len(s.vertices), 4), uint8)
            vcolors[:] = s.color
        v2a = s.vertex_to_atom_map()
        if v2a is None:
            if amask.all():
                v = slice(len(vcolors))
            else:
                session.logger.info('No atom associations for surface #%s' % s.id_string)
                continue
        else:
            v = amask[v2a]
        vcolors[v] = func(vcolors[v])
        s.vertex_colors = vcolors

    # TODO:
    # # Handle surface models specified without specifying atoms
    # from chimerax.atomic import MolecularSurface, Structure
    # from chimerax.map import Volume
    # osurfs = []
    # for s in objects.models:
    #     if isinstance(s, MolecularSurface):
    #         if not s in surfs:
    #             osurfs.append(s)
    #     elif isinstance(s, Volume) or (not isinstance(s, Structure) and not s.empty_drawing()):
    #         osurfs.append(s)
    # for s in osurfs:
    #     s.set_transparency(alpha)
    # surfs.extend(osurfs)

    return surfs

# -----------------------------------------------------------------------------
#
def _change_hue(colors, amount, tag):
    from numpy import amin, amax, around, clip, remainder
    from chimerax.core.colors import rgb_to_hls, hls_to_rgb

    min_rgb = amin(colors[:, 0:3], axis=1)
    max_rgb = amax(colors[:, 0:3], axis=1)
    chromatic_mask = min_rgb != max_rgb

    hls = rgb_to_hls(colors[chromatic_mask, 0:3] / 255.0)
    if tag == 'add':
        hls[:, 0] = remainder(hls[:, 0] + amount, 1)
    elif tag == 'set':
        hls[:, 0] = clip(amount, 0, 1)
    colors[chromatic_mask, 0:3] = clip(around(hls_to_rgb(hls) * 255.0), 0, 255)
    return colors

def _change_saturation(colors, amount, tag):
    from numpy import amin, amax, around, clip
    from chimerax.core.colors import rgb_to_hls, hls_to_rgb

    min_rgb = amin(colors[:, 0:3], axis=1)
    max_rgb = amax(colors[:, 0:3], axis=1)
    chromatic_mask = min_rgb != max_rgb

    hls = rgb_to_hls(colors[chromatic_mask, 0:3] / 255.0)
    if tag == 'add':
        hls[:, 2] = clip(hls[:, 2] + amount, 0, 1)
    elif tag == 'mul':
        hls[:, 2] = clip(hls[:, 2] * amount, 0, 1)
    elif tag == 'set':
        hls[:, 2] = clip(amount, 0, 1)
    colors[chromatic_mask, 0:3] = clip(around(hls_to_rgb(hls) * 255.0), 0, 255)
    return colors


def _change_lightness(colors, amount, tag):
    from numpy import clip, around
    from chimerax.core.colors import rgb_to_hls, hls_to_rgb

    hls = rgb_to_hls(colors[:, 0:3] / 255.0)
    if tag == 'add':
        hls[:, 1] = clip(hls[:, 1] + amount, 0, 1)
    elif tag == 'mul':
        hls[:, 1] = clip(hls[:, 1] * amount, 0, 1)
    elif tag == 'set':
        hls[:, 1] = clip(amount, 0, 1)
    colors[:, 0:3] = clip(around(hls_to_rgb(hls) * 255.0), 0, 255)
    return colors


def _change_whiteness(colors, amount, tag):
    from numpy import clip, around
    from chimerax.core.colors import rgb_to_hwb, hwb_to_rgb

    hwb = rgb_to_hwb(colors[:, 0:3] / 255.0)
    if tag == 'add':
        hwb[:, 1] = clip(hwb[:, 1] + amount, 0, 1)
    elif tag == 'mul':
        hwb[:, 1] = clip(hwb[:, 1] * amount, 0, 1)
    elif tag == 'set':
        hwb[:, 1] = clip(amount, 0, 1)
    colors[:, 0:3] = clip(around(hwb_to_rgb(hwb) * 255.0), 0, 255)
    return colors


def _change_blackness(colors, amount, tag):
    from numpy import clip, around
    from chimerax.core.colors import rgb_to_hwb, hwb_to_rgb

    hwb = rgb_to_hwb(colors[:, 0:3] / 255.0)
    if tag == 'add':
        hwb[:, 2] = clip(hwb[:, 2] + amount, 0, 1)
    elif tag == 'mul':
        hwb[:, 2] = clip(hwb[:, 2] * amount, 0, 1)
    elif tag == 'set':
        hwb[:, 2] = clip(amount, 0, 1)
    colors[:, 0:3] = clip(around(hwb_to_rgb(hwb) * 255.0), 0, 255)
    return colors


def _change_component(colors, amount, tag, component):
    from numpy import clip, around
    if component == 'red':
        i = 0
    elif component == 'green':
        i = 1
    elif component == 'blue':
        i = 2
    elif component == 'alpha':
        i = 3
    else:
        raise RuntimeError("unknown color component")
    amount *= 255.0
    if tag == 'add':
        colors[:, i] = clip(colors[:, i] + amount, 0, 255)
    elif tag == 'mul':
        colors[:, i] = clip(colors[:, i] * amount, 0, 255)
    elif tag == 'set':
        colors[:, i] = clip(amount, 0, 255)
    return colors


def _rgb_interpolate(colors, amount, to):
    colors[:, 0:3] = colors[:, 0:3] + amount * (to - colors[:, 0:3])
    return colors

def _hwb_contrast(orig_hwb, orig_luminance, wb, amount, _count=10):
    from chimerax.core.colors import rgb_to_hwb, hwb_to_rgb, luminance
    w, b = wb
    MIN_CONTRAST = 4.5
    # iterate solution by computing luminance and contrast ratio
    from numpy import empty, zeros, ones
    hwb = empty(orig_hwb.shape)
    left = zeros(len(orig_hwb))
    right = ones(len(orig_hwb))
    for _ in range(_count):
        f = (left + right) / 2
        hwb[:, 0] = orig_hwb[:, 0]
        hwb[:, 1] = orig_hwb[:, 1] + f * (w - orig_hwb[:, 1])
        hwb[:, 2] = orig_hwb[:, 2] + f * (b - orig_hwb[:, 2])
        lumin = luminance(hwb_to_rgb(hwb))
        if w:
            # adding white, luminance should be larger
            contrast_ratio = (lumin + 0.05) / (orig_luminance + 0.05)
        else:
            contrast_ratio = (orig_luminance + 0.05) / (lumin + 0.05)
        smaller = contrast_ratio < MIN_CONTRAST
        left[smaller] = f[smaller]
        greater = contrast_ratio > MIN_CONTRAST
        right[greater] = f[greater]
    # blend by amount
    hwb[:, 1] += amount * (w - hwb[:, 1])
    hwb[:, 2] += amount * (b - hwb[:, 2])
    return hwb


def _constrast(colors, amount):
    from chimerax.core.colors import rgb_to_hwb, hwb_to_rgb, luminance
    from numpy import empty, logical_and, zeros
    from numpy import clip, around
    rgb = colors[:, 0:3] / 255
    lumin = luminance(rgb)
    black_max = lumin >= .5
    white_max = lumin < .5
    if amount == 1:
        colors[black_max, 0:3] = (0, 0, 0)
        colors[white_max, 0:3] = (255, 255, 255)
        return colors
    # find minimum contrast color
    # and linearly interplate between minimum and maximum constrast color
    # minimum constrast color needs a binary search as per
    # https://www.w3.org/TR/css-color-4/#contrast-adjuster
    hwb_contrast_color = empty((len(colors), 3))
    contrast_ratio = empty((len(colors),))
    contrast_ratio[black_max] = (lumin[black_max] + 0.05) / 0.05
    contrast_ratio[white_max] = 1.05 / (lumin[white_max] + 0.05)
    
    # if the contrast ratio is less than the minimum constrast,
    # set the contrast color to the maximum contrast color
    MIN_CONTRAST = 4.5
    mask = zeros((len(colors),), dtype=bool)
    mask[white_max] = logical_and(white_max[white_max], contrast_ratio[white_max] <= MIN_CONTRAST)
    hwb_contrast_color[mask] = (0, 1, 0)
    mask = zeros((len(colors),), dtype=bool)
    mask[black_max] = logical_and(black_max[black_max], contrast_ratio[black_max] <= MIN_CONTRAST)
    hwb_contrast_color[mask] = (0, 0, 1)
    # limit white_max and black_max to ones where we need to compute
    # the minimum contrast color
    white_max[white_max] = logical_and(white_max[white_max], contrast_ratio[white_max] > MIN_CONTRAST)
    black_max[black_max] = logical_and(black_max[black_max], contrast_ratio[black_max] > MIN_CONTRAST)
    hwb = rgb_to_hwb(rgb[white_max])
    hwb_contrast_color[white_max] = _hwb_contrast(hwb, lumin[white_max], (1, 0), amount)
    hwb = rgb_to_hwb(rgb[black_max])
    hwb_contrast_color[black_max] = _hwb_contrast(hwb, lumin[black_max], (0, 1), amount)
    colors[:, 0:3] = clip(around(hwb_to_rgb(hwb_contrast_color) * 255.0), 0, 255)
    return colors


# leave out 'tint', 'shade', and 'alpha' to avoid having to document them
ADJUST_TYPES = ('hue', 'saturation', 'lightness', 'whiteness', 'blackness', 'contrast', 'red', 'green', 'blue')
OP_TYPES = ('+', '-', '*')


def color_modify(session, objects, adjuster, op, number=None, what=None, target=None):
    from chimerax.core.errors import UserError
    if adjuster == 'hue':
        amount = number / 360
        if op == '+':
            color_func(session, objects, what, target,
                     lambda c, a=amount: _change_hue(c, a, 'add'),
                     "Changed hue of", "color hue")
        elif op == '-':
            color_func(session, objects, what, target,
                     lambda c, a=-amount: _change_hue(c, a, 'add'),
                     "Changed hue of", "color hue")
        elif op == '*':
            raise UserError("Unable to scale hue")
        else:  # op == None
            color_func(session, objects, what, target,
                     lambda c, a=amount: _change_hue(c, a, 'set'),
                     "Set hue of", "color hue")
        return
    if adjuster == 'contrast' and number is None:
        percentage = 100
    elif number is None:
        raise UserError('Missing percentage')
    amount = number / 100
    if adjuster == 'saturation':
        if op == '+':
            color_func(session, objects, what, target,
                     lambda c, a=amount: _change_saturation(c, a, 'add'),
                     "Saturated", "color saturation")
        elif op == '-':
            color_func(session, objects, what, target,
                     lambda c, a=-amount: _change_saturation(c, a, 'add'),
                     "Desaturated", "color saturation")
        elif op == '*':
            color_func(session, objects, what, target,
                     lambda c, a=amount: _change_saturation(c, a, 'mul'),
                     "Changed saturation of", "color saturation")
        else:  # op == None
            color_func(session, objects, what, target,
                     lambda c, a=amount: _change_saturation(c, a, 'set'),
                     "Set saturation of", "color saturation")
    elif adjuster == 'lightness':
        if op == '+':
            color_func(session, objects, what, target,
                     lambda c, a=amount: _change_lightness(c, a, 'add'),
                     "Lightened", "color lightness")
        elif op == '-':
            color_func(session, objects, what, target,
                     lambda c, a=-amount: _change_lightness(c, a, 'add'),
                     "Darkened", "color lightness")
        elif op == '*':
            color_func(session, objects, what, target,
                     lambda c, a=amount: _change_lightness(c, a, 'mul'),
                     "Changed lightness of", "color lightness")
        else:  # op == None
            color_func(session, objects, what, target,
                     lambda c, a=amount: _change_lightness(c, a, 'set'),
                     "Set lightness of", "color lightness")
    elif adjuster == 'whiteness':
        if op == '+':
            color_func(session, objects, what, target,
                     lambda c, a=amount: _change_whiteness(c, a, 'add'),
                     "Increased hue's whiteness of", "color whiteness")
        elif op == '-':
            color_func(session, objects, what, target,
                     lambda c, a=-amount: _change_whiteness(c, a, 'add'),
                     "Decreased hue's whiteness of", "color whiteness")
        elif op == '*':
            color_func(session, objects, what, target,
                     lambda c, a=amount: _change_whiteness(c, a, 'mul'),
                     "Changed hue's whiteness of", "color whiteness")
        else:  # op == None
            color_func(session, objects, what, target,
                     lambda c, a=amount: _change_whiteness(c, a, 'set'),
                     "Set hue's whiteness of", "color whiteness")
    elif adjuster == 'blackness':
        if op == '+':
            color_func(session, objects, what, target,
                     lambda c, a=amount: _change_blackness(c, a, 'add'),
                     "Increased hue's blackness of", "color blackness")
        elif op == '-':
            color_func(session, objects, what, target,
                     lambda c, a=-amount: _change_blackness(c, a, 'add'),
                     "Decreased hue's blackness of", "color blackness")
        elif op == '*':
            color_func(session, objects, what, target,
                     lambda c, a=amount: _change_blackness(c, a, 'mul'),
                     "Changed hue's blackness of", "color blackness")
        else:  # op == None
            color_func(session, objects, what, target,
                     lambda c, a=amount: _change_blackness(c, a, 'set'),
                     "Set hue's blackness of", "color blackness")
    elif adjuster == 'tint':
        if op is not None:
            raise UserError("No op allowed")
        if 0 <= amount <= 1:
            color_func(session, objects, what, target,
                     lambda c, a=amount: _rgb_interpolate(c, a, (255, 255, 255)),
                     "Tinted", "color tint")
        else:
            raise UserError("percentage must be between 0 and 100 inclusive")
    elif adjuster == 'shade':
        if op is not None:
            raise UserError("No op allowed")
        if 0 <= amount <= 1:
            color_func(session, objects, what, target,
                     lambda c, a=amount: _rgb_interpolate(c, a, (0, 0, 0)),
                     "Shaded", "color shade")
        else:
            raise UserError("percentage must be between 0 and 100 inclusive")
    elif adjuster == 'contrast':
        if op is not None:
            raise UserError("No op allowed")
        if 0 <= amount <= 1:
            color_func(session, objects, what, target,
                     lambda c, a=amount: _constrast(c, a),
                     "Set contrasting color of ", "color contrast")
        else:
            raise UserError("percentage must be between 0 and 100 inclusive")
    elif adjuster in ('red', 'green', 'blue', 'alpha'):
        if op == '+':
            color_func(session, objects, what, target,
                     lambda c, a=amount: _change_component(c, a, 'add', adjuster),
                     "Increased %sness of" % adjuster, "color %s" % adjuster)
        elif op == '-':
            color_func(session, objects, what, target,
                     lambda c, a=-amount: _change_component(c, a, 'add', adjuster),
                     "Decreased %sness of" % adjuster, "color %s" % adjuster)
        elif op == '*':
            color_func(session, objects, what, target,
                     lambda c, a=amount: _change_component(c, a, 'mul', adjuster),
                     "Changed %sness of" % adjuster, "color %s" % adjuster)
        else:  # op == None
            color_func(session, objects, what, target,
                     lambda c, a=amount: _change_component(c, a, 'set', adjuster),
                     "Set %sness of" % adjuster, "color %s" % adjuster)
    else:
        raise UserError("Color \"%s\" not implemented yet" % adjuster)

def color_sequential(session, objects, level='residues', what=None, target=None,
                     palette=None, transparency=None, undo_name="color"):
    '''
    Color a sequence of atomic objects using a color palette.

    objects : Objects
      Which objects to color.
    level : "residues", "chains", "polymers", "structures"
      Assigns each object a color from a palette.  Default "residues".
    what :  'atoms', 'cartoons', 'ribbons', 'surfaces', 'bonds', 'pseudobonds' or None
      What to color. Everything is colored if option is not specified.
    target : string containing letters 'a', 'b', 'c', 'p', 'r', 's'
      Alternative to the "what" option for specifying what to color.
      Characters indicating what to color, a = atoms, c = cartoon, r = cartoon, s = surfaces,
      b = bonds, p = pseudobonds
      Everything is colored if no target is specified.
    palette : :class:`.Colormap`
      Color map to use with sequential coloring.
    transparency : float
      Percent transparency to use.  If not specified current transparency is preserved.
    '''
    if objects is None:
        from chimerax.core.objects import all_objects
        objects = all_objects(session)
    
    try:
        f = _SequentialColor[level]
    except KeyError:
        from chimerax.core.errors import UserError
        raise UserError('sequential "%s" not implemented yet' % level)

    target, is_default_target = get_targets(target, what)

    from chimerax.core.undo import UndoState
    undo_state = UndoState(undo_name)

    opacity = None
    if transparency is not None:
        opacity = min(255, max(0, int(2.56 * (100 - transparency))))

    f(session, objects, palette, opacity, target, undo_state)

    session.undo.register(undo_state)

def _none_possible_colors(item_colors, attr_vals, non_none_colors, no_value_color):
    ci = 0
    colors = []
    import sys
    for item_color, val in zip(item_colors, attr_vals):
        if val is None:
            if no_value_color is None:
                colors.append(item_color)
            else:
                colors.append(no_value_color.uint8x4())
        else:
            colors.append(non_none_colors[ci])
            ci += 1
    import numpy
    return numpy.array(colors, dtype=numpy.uint8)

def color_by_attr(session, attr_name, atoms=None, what=None, target=None, average=None,
                  palette=None, range=None, no_value_color=None,
                  transparency=None, undo_name="color byattribute", key=False,
                  log_info = True):
    '''
    Color atoms by attribute value using a color palette.

    attr_name : string (actual Python attribute name optionally prefixed by 'a:'/'r:'/'m:'
      for atom/residue/model attribute. If no prefix, then the Atom/Residue/Structure classes
      will be searched for the attribute (in that order).
    atoms : Atoms
    what : list of 'atoms', 'cartoons', 'ribbons', 'surface'
      What to color.  Cartoon and ribbon use average bfactor for each residue.
      Default is to color all depictions.
    target : string containing letters 'a', 'b', 'c', 'p', 'r', 's'
      Alternate way to specify what to color allows specifying more than one of atoms (a),
      cartoon (c), ribbon (r), surface (s).
    average : 'residues' or None
      Whether to average attribute over residues.
    palette : :class:`.Colormap`
      Color map to use with sequential coloring.
    range : 2 comma-separated floats or "full"
      Specifies the range of map values used for sampling from a palette.
    transparency : float
      Percent transparency to use.  If not specified current transparency is preserved.
    key : boolean
      Whether to also show a color key.
    log_info: boolean
      Whether to log number of atoms, residues and attribute value range.  Default True.
    '''

    from chimerax.core.errors import UserError
    from chimerax.atomic import Atom, Residue, Structure
    from .defattr import parse_attribute_name
    attr_name, class_obj = parse_attribute_name(session, attr_name, allowable_types=[int, float])

    if atoms is None:
        from chimerax.atomic import all_atoms
        atoms = all_atoms(session)

    if len(atoms) == 0:
        session.logger.warning('No atoms specified')
        return
        
    target, is_default_target = get_targets(target, what)

    from chimerax.core.undo import UndoState
    undo_state = UndoState(undo_name)
        
    opacity = None
    if transparency is not None:
        opacity = min(255, max(0, int(2.56 * (100 - transparency))))

    if class_obj == Atom:
        attr_objs = atoms
    elif class_obj == Residue:
        attr_objs = atoms.residues
    else:
        attr_objs = atoms.structures
    from chimerax.core.commands import plural_of
    attr_names = plural_of(attr_name)
    needs_none_processing = True
    attr_vals = None
    if hasattr(attr_objs, attr_names):
        # attribute found in Collection; try to maximize efficiency
        needs_none_processing = False
        if average == 'residues' and class_obj == Atom:
            residues = atoms.unique_residues
            res_average = { r: getattr(r.atoms, attr_names).mean() for r in residues }
            attr_vals = [res_average[r] for r in atoms.residues]
        else:
            import numpy
            attr_vals = getattr(attr_objs, attr_names)
            if not isinstance(attr_vals, numpy.ndarray):
                # might have Nones
                needs_none_processing = True
        if not needs_none_processing:
            acolors = _value_colors(palette, range, attr_vals)
            if 'c' in target or 'f' in target:
                if class_obj == Atom:
                    if average != 'residues':
                        # these vars already computed if average == 'residues'...
                        residues = atoms.unique_residues
                        res_average = { r: getattr(r.atoms, attr_names).mean() for r in residues }
                    res_attr_vals = [res_average[r] for r in residues]
                else:
                    residues = atoms.unique_residues
                    if class_obj == Residue:
                        res_attr_vals = getattr(residues, attr_names)
                    else:
                        res_attr_vals = getattr(residues.structures, attr_names)
                rib_colors = ring_colors = _value_colors(palette, range, res_attr_vals)
    if needs_none_processing:
        if attr_vals is None:
            attr_vals = [getattr(o, attr_name, None) for o in attr_objs]
        has_none = None in attr_vals
        if has_none:
            if average == 'residues' and class_obj == Atom:
                residues = atoms.unique_residues
                res_average = {}
                for r in residues:
                    vals = []
                    for a in r.atoms:
                        val = getattr(a, attr_name, None)
                        if val is not None:
                            vals.append(val)
                    res_average[r] = sum(vals)/len(vals) if vals else None
                attr_vals = [res_average[r] for r in atoms.residues]
            non_none_attr_vals = [v for v in attr_vals if v is not None]
            if non_none_attr_vals:
                non_none_colors = _value_colors(palette, range, non_none_attr_vals)
            else:
                non_none_colors = None
                session.logger.warning("All '%s' values are None" % attr_name)
            acolors = _none_possible_colors(atoms.colors, attr_vals, non_none_colors, no_value_color)
            if 'c' in target or 'f' in target:
                if class_obj == Atom:
                    if average != 'residues':
                        # these vars already computed if average == 'residues'...
                        residues = atoms.unique_residues
                        res_average = {}
                        for r in residues:
                            vals = []
                            for a in r.atoms:
                                val = getattr(a, attr_name, None)
                                if val is not None:
                                    vals.append(val)
                            res_average[r] = sum(vals)/len(vals) if vals else None
                    res_attr_vals = [res_average[r] for r in residues]
                else:
                    residues = atoms.unique_residues
                    if class_obj == Residue:
                        res_attr_vals = [getattr(r, attr_name, None) for r in residues]
                    else:
                        res_attr_vals = [getattr(r.structure, attr_name, None) for r in residues]
                non_none_res_attr_vals = [v for v in res_attr_vals if v is not None]
                if non_none_res_attr_vals:
                    non_none_res_colors = _value_colors(palette, range, non_none_res_attr_vals)
                else:
                    non_none_res_colors = None
                rib_colors = _none_possible_colors(residues.ribbon_colors, res_attr_vals,
                    non_none_res_colors, no_value_color)
                ring_colors = _none_possible_colors(residues.ring_colors, res_attr_vals,
                    non_none_res_colors, no_value_color)
            # for later min/max message...
            attr_vals = non_none_attr_vals
        else:
            if average == 'residues' and class_obj == Atom:
                residues = atoms.unique_residues
                res_average = { r: sum([getattr(a, attr_name) for a in r.atoms])/r.num_atoms for r in residues }
                attr_vals = [res_average[r] for r in atoms.residues]
            acolors = _value_colors(palette, range, attr_vals)
            if 'c' in target or 'f' in target:
                if class_obj == Atom:
                    if average != 'residues':
                        # these vars already computed if average == 'residues'...
                        residues = atoms.unique_residues
                        res_average = { r: sum([getattr(a, attr_name)
                            for a in r.atoms])/r.num_atoms for r in residues }
                    res_attr_vals = [res_average[r] for r in residues]
                else:
                    residues = atoms.unique_residues
                    if class_obj == Residue:
                        res_attr_vals = [getattr(r, attr_name, None) for r in residues]
                    else:
                        res_attr_vals = [getattr(r.structure, attr_name) for r in residues]
                rib_colors = ring_colors = _value_colors(palette, range, res_attr_vals)

    acolors[:, 3] = atoms.colors[:, 3] if opacity is None else opacity
    msg = []
    if 'a' in target:
        undo_state.add(atoms, "colors", atoms.colors, acolors)
        atoms.colors = acolors
        msg.append('%d atoms' % len(atoms))

    if 'c' in target:
        rib_colors[:, 3] = residues.ribbon_colors[:, 3] if opacity is None else opacity
        undo_state.add(residues, "ribbon_colors", residues.ribbon_colors, rib_colors)
        residues.ribbon_colors = rib_colors
        msg.append('%d residues' % len(residues))

    if 'f' in target:
        ring_colors[:, 3] = residues.ring_colors[:, 3] if opacity is None else opacity
        undo_state.add(residues, "ring_colors", residues.ring_colors, ring_colors)
        residues.ring_colors = ring_colors
        # TODO: msg.append('%d residues' % len(residues))

    if 's' in target:
        ns = _color_surfaces_at_atoms(atoms, per_atom_colors = acolors, opacity = opacity,
                                      undo_state=undo_state)
        if ns > 0:
            msg.append('%d surfaces' % ns)

    session.undo.register(undo_state)
    if len(attr_vals):
        min_val, max_val, cmap = _value_colors(palette, range, attr_vals, return_cmap_data=True)
        if key:
            from chimerax.color_key import show_key
            show_key(session, cmap)

        if msg:
            r = 'atom %s range' if average is None else 'residue average %s range'
            m = ', '.join(msg) + ', %s %.3g to %.3g' % (r % attr_name, min_val, max_val)
            if log_info:
                session.logger.status(m, log=True)

# -----------------------------------------------------------------------------
#
def _color_surfaces_at_atoms(atoms = None, color = None, per_atom_colors = None,
                             opacity = None, undo_state = None):
    from chimerax import atomic
    surfs = atomic.surfaces_with_atoms(atoms)
    for s in surfs:
        if undo_state and hasattr(s, 'color_undo_state'):
            colors_before = s.color_undo_state
            s.color_atom_patches(atoms, color, per_atom_colors, opacity = opacity)
            undo_state.add(s, 'color_undo_state', colors_before, s.color_undo_state)
        else:
            s.color_atom_patches(atoms, color, per_atom_colors, opacity = opacity)
    return len(surfs)

# -----------------------------------------------------------------------------
#
def _color_surfaces_at_residues(residues, colors, opacity = None, undo_state = None):
    atoms, acolors = _residue_atoms_and_colors(residues, colors)
    num_surf = _color_surfaces_at_atoms(atoms, per_atom_colors = acolors, opacity=opacity,
                                        undo_state=undo_state)
    return num_surf

# -----------------------------------------------------------------------------
#
def _residue_atoms_and_colors(residues, colors):
    atoms = residues.atoms
    from numpy import repeat
    acolors = repeat(colors, residues.num_atoms, axis=0)
    return atoms, acolors

def _value_colors(palette, range, values, *, return_cmap_data=False):
    from chimerax.surface.colorvol import _use_full_range, _colormap_with_range
    min_val, max_val = min(values), max(values)
    r = (min_val, max_val) if _use_full_range(range, palette) else range
    if r is not None and r[0] == r[1]:
        # all values the same; artificially manipulate the range to get the
        # 'middle' of the color range used
        r = (r[0]-1, r[1]+1)
    cmap = _colormap_with_range(palette, r, default = 'blue-white-red')
    if return_cmap_data:
        return min_val, max_val, cmap
    colors = cmap.interpolated_rgba8(values)
    return colors
        
def color_zone(session, surfaces, near, distance=2, sharp_edges = False,
               bond_point_spacing = None, far_color = None, update = True, undo_state = None):
    '''
    Color surfaces to match nearby atom colors.

    surfaces : list of models
      Surfaces to color.
    near : Atoms
      Color surfaces to match closest atom within specified zone distance.
    distance : float
      Zone distance used with zone option.
    sharp_edges : bool
      If true change the surface to add cut lines exactly at color zone the boundaries. This makes sharp
      color transitions at the boundaries between different color patches.  If false, or zone option is not
      used then the surface is not changed.
    bond_point_spacing : float
      Include points along bonds between the given atoms at this spacing.
    far_color : Color or 'keep' or None
      Color surface points beyond the distance range this color.  If None then far
      points are given the current surface single color.  If 'keep' then far points
      keep their current color.  Default is None.
    update : bool
      Whether to update surface color when surface shape changes.  Default true.
    '''
    if len(surfaces) == 0:
        from chimerax.core.errors import UserError
        raise UserError('color zone: No surfaces specified.')
    if len(near) == 0:
        from chimerax.core.errors import UserError
        raise UserError('color zone: No atoms specified.')
    atoms = near
    bonds = near.intra_bonds if bond_point_spacing is not None else None
    from chimerax.surface.colorzone import points_and_colors, color_zone, color_zone_sharp_edges
    points, colors = points_and_colors(atoms, bonds, bond_point_spacing)
    from chimerax.core.colors import Color
    fcolor = far_color.uint8x4() if isinstance(far_color, Color) else far_color
    from chimerax.core.undo import UndoState
    undo_state = UndoState('color zone')
    for s in surfaces:
        cprev = s.color_undo_state
        # Transform points to surface coordinates
        tf = s.scene_position
        spoints = points if tf.is_identity() else (tf.inverse() * points)
        color_zone(s, spoints, colors, distance, sharp_edges = sharp_edges,
                   far_color = fcolor, auto_update = update)
        undo_state.add(s, 'color_undo_state', cprev, s.color_undo_state)

    session.undo.register(undo_state)

        
def color_single(session, models = None):
    '''
    Turn off per-vertex coloring for specified models.

    models : list of models
    '''
    from chimerax.core.undo import UndoState
    undo_state = UndoState('color single')

    if models is None:
        models = session.models.list()

    # Save undo state before setting any model single colors
    # since setting may change child model colors, e.g. with Volume.
    for m in models:
        if m.vertex_colors is not None:
            if m.auto_recolor_vertices is not None:
                undo_state.add(m, 'auto_recolor_vertices', m.auto_recolor_vertices, None)
            undo_state.add(m, 'vertex_colors', m.vertex_colors, None)
            m.vertex_colors = None

    session.undo.register(undo_state)
    
from chimerax.core.commands import StringArg
class TargetArg(StringArg):
    """String containing characters indicating what to color:
    a = atoms, c = cartoon, r = cartoon, s = surfaces, b = bonds, p = pseudobonds, f = (filled) rings,
    m = models
    """
    name = "characters from 'abcfmprs'"

    @staticmethod
    def parse(text, session):
        if not text:
            from chimerax.core.commands import AnnotationError
            raise AnnotationError("Expected %s" % TargetArg.name)
        from chimerax.core.commands import next_token
        token, text, rest = next_token(text)
        for c in token:
            if not c in ALL_TARGETS:
                from chimerax.core.commands import AnnotationError
                raise AnnotationError("Character '%s' is not an allowed target, must be one of %s"
                                      % (c, ALL_TARGETS))
        return token, text, rest

# -----------------------------------------------------------------------------
#
def register_command(logger):
    from chimerax.core.commands import register, CmdDesc, ColorArg, ColormapArg, ColormapRangeArg
    from chimerax.core.commands import ObjectsArg, ModelsArg, create_alias, EmptyArg, Or, EnumOf
    from chimerax.core.commands import ListOf, FloatArg, BoolArg, SurfacesArg, StringArg, Color8Arg
    from chimerax.core.commands import create_alias
    from chimerax.atomic import AtomsArg
    what_arg = ListOf(EnumOf((*WHAT_TARGETS.keys(),)))
    desc = CmdDesc(required=[('objects', Or(ObjectsArg, EmptyArg))],
                   optional=[('color', Or(ColorArg, EnumOf(_SpecialColors))),
                             ('what', what_arg)],
                   keyword=[('target', TargetArg),
                            ('transparency', FloatArg),
                            ('halfbond', BoolArg),
                   ],
                   synopsis="color objects")
    register('color', desc, color, logger=logger)
    create_alias('colour', 'color $*', logger=logger, url="help:user/commands/color.html")

    # color modify
    adjust_arg = EnumOf(ADJUST_TYPES)
    op_arg = EnumOf(OP_TYPES)
    desc = CmdDesc(required=[('objects', Or(ObjectsArg, EmptyArg)),
                             ('adjuster', adjust_arg),
                             ('op', Or(op_arg, EmptyArg)),
                             ('number', Or(FloatArg, EmptyArg))],
                   optional=[('what', what_arg)],
                   keyword=[('target', TargetArg)],
                   synopsis="saturate color")
    register('color modify', desc, color_modify, logger=logger)

    # color a sequence of atomic objects
    desc = CmdDesc(required=[('objects', Or(ObjectsArg, EmptyArg))],
                   optional=[('level', EnumOf(_SequentialLevels))],
                   keyword=[('target', TargetArg),
                            ('what', what_arg),
                            ('palette', ColormapArg),
                            ('transparency', FloatArg),
                       ],
                   synopsis="color a sequence of atomic objects using a palette")
    register('color sequential', desc, color_sequential, logger=logger)

    # color atoms by attribute
    desc = CmdDesc(required=[('attr_name', StringArg),
                            ('atoms', Or(AtomsArg, EmptyArg))],
                   optional=[('what', ListOf(EnumOf(('atoms', 'cartoons', 'ribbons', 'surfaces'))))],
                   keyword=[('target', TargetArg),
                            ('average', EnumOf(('residues',))),
                            ('palette', ColormapArg),
                            ('range', ColormapRangeArg),
                            ('no_value_color', ColorArg),
                            ('transparency', FloatArg),
                            ('key', BoolArg),
                            ('log_info', BoolArg)],
                   synopsis="color atoms by bfactor")
    register('color byattribute', desc, color_by_attr, logger=logger)
    create_alias('color bfactor', 'color byattribute bfactor $*', logger=logger,
            url="help:user/commands/color.html#byattribute")

    # color by nearby atoms
    desc = CmdDesc(required=[('surfaces', SurfacesArg)],
                   keyword=[('near', AtomsArg),
                            ('distance', FloatArg),
                            ('sharp_edges', BoolArg),
                            ('bond_point_spacing', FloatArg),
                            ('far_color', Or(EnumOf(['keep']), ColorArg)),
                            ('update', BoolArg),
                       ],
                   required_arguments = ['near'],
                   synopsis="color surfaces to match nearby atoms")
    register('color zone', desc, color_zone, logger=logger)

    # color a single color
    desc = CmdDesc(optional=[('models', ModelsArg)],
                   synopsis="turn off model per-vertex coloring")
    register('color single', desc, color_single, logger=logger)
