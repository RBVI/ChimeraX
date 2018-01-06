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

_SpecialColors = ["byatom", "byelement", "byhetero", "bychain", "bypolymer", "bynucleotide", "bymodel",
                  "fromatoms", "random"]

_SequentialLevels = ["residues", "chains", "polymers", "structures"]
# More possible sequential levels: "helix", "helices", "strands", "SSEs", "volmodels", "allmodels"

DEFAULT_TARGETS = 'acslbpd'
ALL_TARGETS = 'acslbpd'
WHAT_TARGETS = {
    'atoms': 'a',
    'cartoons': 'c', 'ribbons': 'c',
    'surfaces': 's',
    'labels': 'l',
    'bonds': 'b',
    'pseudobonds': 'p',
    # 'distances': 'd',  # TODO: conflicts with distance argument
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


def color(session, objects, color=None, what=None,
          target=None, transparency=None,
          sequential=None, palette=None, halfbond=None,
          map=None, range=None, offset=0, zone=None, distance=2,
          undo_name="color"):
    """Color atoms, ribbons, surfaces, ....

    Parameters
    ----------
    objects : Objects
      Which objects to color.
    color : Color
      Color can be a standard color name or "byatom", "byelement", "byhetero", "bychain", "bypolymer", "bynucleotide", "bymodel".
    what :  'atoms', 'cartoons', 'ribbons', 'surfaces', 'bonds', 'pseudobonds' or None
      What to color. Everything is colored if option is not specified.
    target : string
      Alternative to the "what" option for specifying what to color.
      Characters indicating what to color, a = atoms, c = cartoon, r = cartoon, s = surfaces,
      l = labels, b = bonds, p = pseudobonds, d = distances.
      Everything is colored if no target is specified.
    transparency : float
      Percent transparency to use.  If not specified current transparency is preserved.
    sequential : "residues", "chains", "polymers", "structures"
      Assigns each object a color from a color map.
    palette : :class:`.Colormap`
      Color map to use with sequential coloring.
    halfbond : bool
      Whether to color each half of a bond to match the connected atoms.
      If halfbond is false the bond is given the single color assigned to the bond.
    map : Volume
      Color specified surfaces by sampling from this density map using palette, range, and offset options.
    range : 2 comma-separated floats or "full"
      Specifies the range of map values used for sampling from a palette.
    offset : float
      Displacement distance along surface normals for sampling map when using map option.  Default 0.
    zone : Atoms
      Color surfaces to match closest atom within specified zone distance.
    distance : float
      Zone distance used with zone option.
    """
    if objects is None:
        from . import all_objects
        objects = all_objects(session)
    atoms = objects.atoms
    if color == "byhetero":
        atoms = atoms.filter(atoms.element_numbers != 6)

    target, is_default_target = get_targets(target, what)

    from ..undo import UndoState
    undo_state = UndoState(undo_name)

    # Decide whether to set or preserve transparency
    opacity = None
    if transparency is not None:
        opacity = min(255, max(0, int(2.56 * (100 - transparency))))
    if getattr(color, 'explicit_transparency', False):
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

    if sequential is not None:
        try:
            f = _SequentialColor[sequential]
        except KeyError:
            from ..errors import UserError
            raise UserError("sequential \"%s\" not implemented yet"
                            % sequential)
        else:
            f(session, objects, palette, opacity, target, undo_state)
            session.undo.register(undo_state)
            return

    if zone is not None:
        from ..atomic import MolecularSurface, Structure
        slist = [m for m in objects.models
                 if not m.empty_drawing() and not isinstance(m, (Structure, MolecularSurface))]
        for m in objects.models:
            if hasattr(m, 'surface_drawings_for_vertex_coloring'):
                slist.extend(m.surface_drawings_for_vertex_coloring())
        bonds = None
        auto_update = False
        from ..surface.colorzone import points_and_colors, color_zone
        for s in slist:
            points, colors = points_and_colors(zone, bonds)
            # TODO: save undo data
            s.scene_position.inverse().move(points)	# Transform points to surface coordinates
            color_zone(s, points, colors, distance, auto_update)

    what = []

    bgcolor = session.main_view.background_color

    if 'a' in target:
        # atoms/bonds
        if atoms is not None and color is not None:
            _set_atom_colors(atoms, color, opacity, bgcolor, undo_state)
            what.append('%d atoms' % len(atoms))

    if 'l' in target:
        if not is_default_target:
            session.logger.warning('Label colors not supported yet')

    if 's' in target and (color is not None or map is not None):
        # TODO: save undo data
        from ..atomic import MolecularSurface, concatenate, Structure, PseudobondGroup
        msatoms = [m.atoms for m in objects.models
                   if isinstance(m, MolecularSurface) and not m.atoms.intersects(atoms)]
        satoms = concatenate(msatoms + [atoms]) if msatoms else atoms
        if color == "byhetero":
            satoms = satoms.filter(satoms.element_numbers != 6)
        ns = _set_surface_colors(session, satoms, color, opacity, bgcolor,
                                 map, palette, range, offset, undo_state=undo_state)
        # Handle non-molecular surfaces like density maps
        if color not in _SpecialColors:
            mlist = [m for m in objects.models if not isinstance(m, (Structure, MolecularSurface, PseudobondGroup))]
            for m in mlist:
                _set_model_colors(session, m, color, map, opacity, palette, range, offset)
            ns += len(mlist)
        what.append('%d surfaces' % ns)

    if 'c' in target and color is not None:
        residues = atoms.unique_residues
        _set_ribbon_colors(residues, color, opacity, bgcolor, undo_state)
        what.append('%d residues' % len(residues))

    if 'b' in target:
        if color not in _SpecialColors and color is not None:
            bonds = objects.bonds
            if len(bonds) > 0:
                if color not in _SpecialColors:
                    color_array = color.uint8x4()
                    undo_state.add(bonds, "colors", bonds.colors, color_array)
                    bonds.colors = color_array
                    what.append('%d bonds' % len(bonds))

    if 'p' in target:
        if color not in _SpecialColors and color is not None:
            pbonds = objects.pseudobonds
            if len(pbonds) > 0:
                color_array = color.uint8x4()
                undo_state.add(pbonds, "colors", pbonds.colors, color_array)
                pbonds.colors = color_array
                what.append('%d pseudobonds' % len(pbonds))

    if 'd' in target:
        if not is_default_target:
            session.logger.warning('Distances colors not supported yet')

    if not what:
        what.append('nothing')

    from . import cli
    session.logger.status('Colored %s' % cli.commas(what, ' and'))
    session.undo.register(undo_state)


def _computed_atom_colors(atoms, color, opacity, bgcolor):
    if color in ("byatom", "byelement", "byhetero"):
        c = _element_colors(atoms, opacity)
    elif color == "bychain":
        from ..atomic.colors import chain_colors
        c = chain_colors(atoms.residues.chain_ids)
        c[:, 3] = atoms.colors[:, 3] if opacity is None else opacity
    elif color == "bypolymer":
        from ..atomic.colors import polymer_colors
        c = atoms.colors.copy()
        sc,amask = polymer_colors(atoms.residues)
        c[amask,:] = sc[amask,:]
        c[amask, 3] = atoms.colors[amask, 3] if opacity is None else opacity
    elif color == "bynucleotide":
        from ..atomic.colors import nucleotide_colors
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


def _element_colors(atoms, opacity=None):
    from ..atomic.colors import element_colors
    c = element_colors(atoms.element_numbers)
    c[:, 3] = atoms.colors[:, 3] if opacity is None else opacity
    return c


def _set_atom_colors(atoms, color, opacity, bgcolor, undo_state):
    if color in _SpecialColors:
        c = _computed_atom_colors(atoms, color, opacity, bgcolor)
        if c is not None:
            undo_state.add(atoms, "colors", atoms.colors, c)
            atoms.colors = c
    else:
        c = atoms.colors
        c[:, :3] = color.uint8x4()[:3]    # Preserve transparency
        if opacity is not None:
            c[:, 3] = opacity
        undo_state.add(atoms, "colors", atoms.colors, c)
        atoms.colors = c


def _set_ribbon_colors(residues, color, opacity, bgcolor, undo_state):
    if color not in _SpecialColors:
        c = residues.ribbon_colors
        c[:, :3] = color.uint8x4()[:3]    # Preserve transparency
        if opacity is not None:
            c[:, 3] = opacity
        undo_state.add(residues, "ribbon_colors", residues.ribbon_colors, c)
        residues.ribbon_colors = c
    elif color == 'bychain':
        from ..atomic.colors import chain_colors
        c = chain_colors(residues.chain_ids)
        c[:, 3] = residues.ribbon_colors[:, 3] if opacity is None else opacity
        undo_state.add(residues, "ribbon_colors", residues.ribbon_colors, c)
        residues.ribbon_colors = c
    elif color == "bypolymer":
        from ..atomic.colors import polymer_colors
        c,rmask = polymer_colors(residues)
        c[rmask, 3] = residues.ribbon_colors[rmask, 3] if opacity is None else opacity
        masked_residues = residues.filter(rmask)
        undo_state.add(masked_residues, "ribbon_colors", masked_residues.ribbon_colors, c[rmask,:])
        masked_residues.ribbon_colors = c[rmask,:]
    elif color == "bynucleotide":
        from ..atomic.colors import nucleotide_colors
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


def _set_surface_colors(session, atoms, color, opacity, bgcolor=None,
                        map=None, palette=None, range=None, offset=0, undo_state=None):
    # TODO: save undo data
    from .scolor import color_surfaces_at_atoms, color_surfaces_by_map_value
    if color in _SpecialColors:
        if color == 'fromatoms':
            ns = color_surfaces_at_atoms(atoms, opacity=opacity)
        else:
            # Surface colored different from atoms
            c = _computed_atom_colors(atoms, color, opacity, bgcolor)
            ns = color_surfaces_at_atoms(atoms, opacity=opacity, per_atom_colors=c)
            
    elif map:
        ns = color_surfaces_by_map_value(atoms, opacity=opacity, map=map, palette=palette,
                                        range=range, offset=offset)
    else:
        ns = color_surfaces_at_atoms(atoms, color, opacity=opacity)
    return ns

def _set_model_colors(session, m, color, map, opacity, palette, range, offset):
    if map is None:
        c = color.uint8x4()
        if not opacity is None:
            c[3] = opacity
        elif not m.single_color is None:
            c[3] = m.single_color[3]
        m.single_color = c
    else:
        if hasattr(m, 'surface_drawings_for_vertex_coloring'):
            surfs = m.surface_drawings_for_vertex_coloring()
        elif not m.empty_drawing():
            surfs = [m]
        else:
            surfs = []
        from .scolor import color_surface_by_map_value
        for s in surfs:
            color_surface_by_map_value(s, map, palette=palette, range=range,
                                       offset=offset, opacity=opacity)

# -----------------------------------------------------------------------------
# Chain ids in each structure are colored from color map ordered alphabetically.
#
def _set_sequential_chain(session, selected, cmap, opacity, target, undo_state):
    # Organize selected atoms by structure and then chain
    uc = selected.atoms.residues.chains.unique()
    chain_atoms = {}
    for c in uc:
        chain_atoms.setdefault(c.structure, []).append((c.chain_id, c.existing_residues.atoms))
    # Make sure there is a colormap
    if cmap is None:
        from .. import colors
        cmap = colors.BuiltinColormaps["rainbow"]
    # Each structure is colored separately with cmap applied by chain
    import numpy
    from ..colors import Color
    for sl in chain_atoms.values():
        sl.sort(key = lambda ca: ca[0])	# Sort by chain id
        colors = cmap.interpolated_rgba(numpy.linspace(0.0, 1.0, len(sl)))
        for color, (chain_id, atoms) in zip(colors, sl):
            c = Color(color)
            if target is None or 'a' in target:
                _set_atom_colors(atoms, c, opacity, None, undo_state)
            if target is None or 'c' in target:
                res = atoms.unique_residues
                _set_ribbon_colors(res, c, opacity, None, undo_state)
            if target is None or 's' in target:
                _set_surface_colors(session, atoms, c, opacity, undo_state=undo_state)

# ----------------------------------------------------------------------------------
# Polymers (unique sequences) in each structure are colored from color map ordered
# by polymer length.
#
def _set_sequential_polymer(session, objects, cmap, opacity, target, undo_state):
    # Organize atoms by structure and then polymer sequence
    uc = objects.atoms.residues.chains.unique()
    seq_atoms = {}
    for c in uc:
        seq_atoms.setdefault(c.structure, {}).setdefault(c.characters, []).append(c.existing_residues.atoms)
    # Make sure there is a colormap
    if cmap is None:
        from .. import colors
        cmap = colors.BuiltinColormaps["rainbow"]
    # Each structure is colored separately with cmap applied by chain
    import numpy
    from ..colors import Color
    for sl in seq_atoms.values():
        sseq = list(sl.items())
        sseq.sort(key = lambda sa: len(sa[0]))	# Sort by sequence length
        colors = cmap.interpolated_rgba(numpy.linspace(0.0, 1.0, len(sseq)))
        for color, (seq, alist) in zip(colors, sseq):
            c = Color(color)
            for atoms in alist:
                if target is None or 'a' in target:
                    _set_atom_colors(atoms, c, opacity, None, undo_state)
                if target is None or 'c' in target:
                    res = atoms.unique_residues
                    _set_ribbon_colors(res, c, opacity, None, undo_state)
                if target is None or 's' in target:
                    _set_surface_colors(session, atoms, c, opacity, undo_state=undo_state)

# -----------------------------------------------------------------------------
#
def _set_sequential_residue(session, selected, cmap, opacity, target, undo_state):
    # Make sure there is a colormap
    if cmap is None:
        from .. import colors
        cmap = colors.BuiltinColormaps["rainbow"]
    # Get chains and atoms in chains with "by_chain"
    # Each chain is colored separately with cmap applied by residue
    import numpy
    from ..colors import Color
    structure_chain_ids = {}
    for structure, chain_id, atoms in selected.atoms.by_chain:
        try:
            cids = structure_chain_ids[structure]
        except KeyError:
            structure_chain_ids[structure] = cids = set()
        cids.add(chain_id)
    for structure, cids in structure_chain_ids.items():
        for chain in structure.chains:
            if chain.chain_id not in cids:
                continue
            residues = chain.existing_residues
            colors = cmap.interpolated_rgba(numpy.linspace(0.0, 1.0, len(residues)))
            for color, r in zip(colors, residues):
                c = Color(color)
                if target is None or 'a' in target:
                    _set_atom_colors(r.atoms, c, opacity, None, undo_state)
                if target is None or 'c' in target:
                    rgba = c.uint8x4()
                    if opacity is not None:
                        rgba[3] = opacity
                    undo_state.add(r, "ribbon_color", r.ribbon_color, rgba)
                    r.ribbon_color = rgba

# -----------------------------------------------------------------------------
#
def _set_sequential_structures(session, selected, cmap, opacity, target, undo_state):
    # Make sure there is a colormap
    if cmap is None:
        from .. import colors
        cmap = colors.BuiltinColormaps["rainbow"]

    from ..atomic import Structure
    models = list(m for m in selected.models if isinstance(m, Structure))
    models.sort(key = lambda m: m.id)
    if len(models) == 0:
        return

    # Each structure is colored separately with cmap applied by chain
    import numpy
    from ..colors import Color
    colors = cmap.interpolated_rgba(numpy.linspace(0.0, 1.0, len(models)))
    for color, m in zip(colors, models):
        c = Color(color)
        if 'a' in target:
            _set_atom_colors(m.atoms, c, opacity, None, undo_state)
        if 'c' in target:
            _set_ribbon_colors(m.residues, c, opacity, None, undo_state)
        if 's' in target:
            # TODO: save surface undo data
            from .scolor import color_surfaces_at_atoms
            color_surfaces_at_atoms(m.atoms, c)

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
        from . import all_objects
        objects = all_objects(session)
    atoms = objects.atoms

    from ..undo import UndoState
    undo_state = UndoState(undo_name)

    target, is_default_target = get_targets(target, what)

    what = []

    if 'a' in target:
        # atoms/bonds
        c = func(atoms.colors)
        undo_state.add(atoms, "colors", atoms.colors, c)
        atoms.colors = c
        what.append('%d atoms' % len(atoms))

    if 'l' in target:
        if not is_default_target:
            session.logger.warning('Label colors not supported yet')

    if 's' in target:
        surfs = _set_surface_color_func(atoms, objects, session, func, undo_state)
        what.append('%d surfaces' % len(surfs))

    if 'c' in target:
        residues = atoms.unique_residues
        c = func(residues.ribbon_colors)
        undo_state.add(residues, "ribbon_colors", residues.ribbon_colors, c)
        residues.ribbon_colors = c
        what.append('%d residues' % len(residues))

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
            what.append('%d pseudobonds' % len(bonds))

    if 'd' in target:
        if not is_default_target:
            session.logger.warning('Distance colors not supported yet')

    if not what:
        what.append('nothing')

    from . import cli
    session.logger.status('%s %s' % (func_text, cli.commas(what, ' and')))
    session.undo.register(undo_state)

def _set_surface_color_func(atoms, objects, session, func, undo_state=None):
    # TODO: save undo data

    # Handle surfaces for specified atoms
    from .. import atomic
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
                session.logger.info('No atom associations for surface #%s'
                                    % s.id_string())
                continue
        else:
            v = amask[v2a]
        vcolors[v] = func(vcolors[v])
        s.vertex_colors = vcolors

    # TODO:
    # # Handle surface models specified without specifying atoms
    # from ..atomic import MolecularSurface, Structure
    # from ..map import Volume
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
def _change_saturation(colors, amount, tag):
    from numpy import amin, amax, around, clip
    from ..colors import rgb_to_hls, hls_to_rgb

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
    from ..colors import rgb_to_hls, hls_to_rgb

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
    from ..colors import rgb_to_hwb, hwb_to_rgb

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
    from ..colors import rgb_to_hwb, hwb_to_rgb

    hwb = rgb_to_hwb(colors[:, 0:3] / 255.0)
    if tag == 'add':
        hwb[:, 2] = clip(hwb[:, 2] + amount, 0, 1)
    elif tag == 'mul':
        hwb[:, 2] = clip(hwb[:, 2] * amount, 0, 1)
    elif tag == 'set':
        hwb[:, 2] = clip(amount, 0, 1)
    colors[:, 0:3] = clip(around(hwb_to_rgb(hwb) * 255.0), 0, 255)
    return colors


def _rgb_interpolate(colors, amount, to):
    colors[:, 0:3] = colors[:, 0:3] + amount * (to - colors[:, 0:3])
    return colors

def _hwb_contrast(orig_hwb, orig_luminance, wb, amount, _count=10):
    from ..colors import rgb_to_hwb, hwb_to_rgb, luminance
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
    from ..colors import rgb_to_hwb, hwb_to_rgb, luminance
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


ADJUST_TYPES = ('saturation', 'lightness', 'whiteness', 'blackness', 'tint', 'shade', 'contrast')
OP_TYPES = ('+', '-', '*')


def color_modify(session, objects, adjuster, op, percentage=None, what=None, target=None):
    from ..errors import UserError
    if adjuster == 'contrast' and percentage is None:
        percentage = 100
    elif percentage is None:
        raise UserError('Missing percentage')
    amount = percentage / 100
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
    else:
        raise UserError("Color \"%s\" not implemented yet" % adjuster)


# -----------------------------------------------------------------------------
#
def register_command(session):
    from . import register, CmdDesc, ColorArg, ColormapArg, ColormapRangeArg, ObjectsArg, create_alias
    from . import EmptyArg, Or, EnumOf, StringArg, ListOf, FloatArg, BoolArg, AtomsArg
    from ..map import MapArg
    what_arg = ListOf(EnumOf((*WHAT_TARGETS.keys(),)))
    desc = CmdDesc(required=[('objects', Or(ObjectsArg, EmptyArg))],
                   optional=[('color', Or(ColorArg, EnumOf(_SpecialColors))),
                             ('what', what_arg)],
                   keyword=[('target', StringArg),
                            ('transparency', FloatArg),
                            ('sequential', EnumOf(_SequentialLevels)),
                            ('halfbond', BoolArg),
                            ('map', MapArg),
                            ('palette', ColormapArg),
                            ('range', ColormapRangeArg),
                            ('offset', FloatArg),
                            ('zone', AtomsArg),
                            ('distance', FloatArg),
                   ],
                   synopsis="color objects")
    register('color', desc, color, logger=session.logger)
    create_alias('colour', 'color $*', logger=session.logger)
    adjust_arg = EnumOf(ADJUST_TYPES)
    op_arg = EnumOf(OP_TYPES)
    desc = CmdDesc(required=[('objects', Or(ObjectsArg, EmptyArg)),
                             ('adjuster', adjust_arg),
                             ('op', Or(op_arg, EmptyArg)),
                             ('percentage', Or(FloatArg, EmptyArg))],
                   optional=[('what', what_arg)],
                   keyword=[('target', StringArg)],
                   synopsis="saturate color")
    register('color modify', desc, color_modify, logger=session.logger)
