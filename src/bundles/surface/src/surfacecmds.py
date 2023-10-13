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

# -------------------------------------------------------------------------------------
#
def surface(session, atoms = None, enclose = None, include = None,
            probe_radius = None, grid_spacing = None, resolution = None, level = None,
            color = None, transparency = None, visible_patches = None,
            sharp_boundaries = None, nthread = None, replace = True, update = True):
    '''
    Compute and display solvent excluded molecular surfaces.

    Parameters
    ----------
    atoms : Atoms
      Atoms controlling which surface patch is shown.
      Surfaces are computed for each chain these atoms belong to and patches
      of these surfaces near the specifed atoms are shown.  Solvent, ligands
      and ions are excluded from each chain surface.
    enclose : Atoms
      Make a surface enclosing exactly these specified atoms excluding
      solvent, ligands and ions
    include : Atoms
      Solvent, ligands or ions to include in the surface.
    probe_radius : float
      Radius of probe sphere rolled over atoms to produce surface.
      Only used for solvent excluded surfaces.  Default is 1.4 Angstroms.
    grid_spacing : float
      Surface is computed on 3-dimensional grid with this spacing
      between grid points along each axis.
    resolution : float
      Specifying a resolution value (Angstroms) causes the surface calculation
      to use a contour surface of a 3-d grid of a sum of Gaussians one centered
      at each atom instead of a solvent excluded surface.  See the molmap command
      for details.  A resolution of 0 computes an SES surface.
    level : float
      Contour level for Gaussian surface in atomic number units.  Each Gaussian has
      height scaled by the atom atomic number.
    color : Color
      Colors surfaces using this color.
    transparency : float
      Percentage transparency for surfaces.
    visible_patches : int
      Maximum number of connected surface pieces per chain to show.
    sharp_boundaries : bool
      Make the surface triangulation have edges exactly between atoms
      so per-atom surface colors and surface patches have smoother edges.
    nthread : int
      Number of CPU threads to use in computing surfaces.
    replace : bool
      Whether to replace an existing surface for the same atoms or make a copy.
    update : bool
      Whether to automatically recompute the surface when atom positions change.  Default True.
    '''

    if resolution is not None and probe_radius is not None:
        session.logger.warning('surface: Can only use probeRadius or resolution,'
                               ' not both, ignoring probeRadius')
        
    from chimerax.atomic.molsurf import MolecularSurface, remove_solvent_ligands_ions
    from chimerax.atomic.molsurf import surface_initial_color, update_color, surfaces_overlapping_atoms

    if replace:
        all_surfs = dict((s.atoms.hash(), s) for s in session.models.list(type = MolecularSurface))
    else:
        all_surfs = {}

    # Set default parameters for new molecular surfaces for probe radius, grid spacing, and sharp boundaries.
    probe = 1.4 if probe_radius is None else probe_radius
        
    if grid_spacing is None:
        grid = 0.5 if resolution is None or resolution <= 0 else 0.1 * resolution
    else:
        grid = grid_spacing
    gridsp = grid_spacing if resolution is None else grid

    if sharp_boundaries is None:
        sharp = True if resolution is None else False
    else:
        sharp = sharp_boundaries

    surfs = []
    new_surfs = []
    if enclose is None:
        atoms = check_atoms(atoms, session) # Warn if no atoms specifed
        num_structure_atoms = sum([s.num_atoms for s in atoms.unique_structures], 0)
        full_structures = (len(atoms) == num_structure_atoms)
        atoms, all_small = remove_solvent_ligands_ions(atoms, include)
        for m, chain_id, show_atoms in atoms.by_chain:
            if all_small:
                enclose_atoms = show_atoms
            else:
                # This code is handling the case that the specified atoms don't contain
                # all of the chain atoms, otherwise we could just use show_atoms.
                if full_structures:
                    chain_atoms = show_atoms
                else:
                    # May have only some of the chain atoms, get all of them.
                    # This is very slow for 863 chains, pdb 5y6p, bug #3908.
                    matoms = m.atoms
                    chain_atoms = matoms.filter(matoms.chain_ids == chain_id)
                enclose_atoms = remove_solvent_ligands_ions(chain_atoms, include)[0]
            s = all_surfs.get(enclose_atoms.hash())
            if s is None:
                stype = 'SES' if resolution is None else 'Gaussian'
                name = '%s_%s %s surface' % (m.name, chain_id, stype)
                rgba = surface_initial_color(color, transparency, enclose_atoms)
                s = MolecularSurface(session, enclose_atoms, show_atoms,
                                     probe, grid, resolution, level,
                                     name, rgba, visible_patches, sharp)
                new_surfs.append((s,m))
            else:
                s.new_parameters(show_atoms, probe_radius, gridsp,
                                 resolution, level, visible_patches, sharp_boundaries)
                update_color(s, color, transparency)
            surfs.append(s)
    else:
        enclose_atoms, eall_small = remove_solvent_ligands_ions(enclose, include)
        if len(enclose_atoms) == 0:
            from chimerax.core.errors import UserError
            raise UserError('No atoms specified by %s' % (enclose.spec,))
        show_atoms = enclose_atoms if atoms is None else atoms.intersect(enclose_atoms)
        s = all_surfs.get(enclose_atoms.hash())
        if s is None:
            mols = enclose.unique_structures
            parent = mols[0] if len(mols) == 1 else None
            name = 'Surface %s' % enclose.spec
            rgba = surface_initial_color(color, transparency, enclose_atoms)
            s = MolecularSurface(session, enclose_atoms, show_atoms,
                                 probe, grid, resolution, level,
                                 name, rgba, visible_patches, sharp)
            new_surfs.append((s,parent))
        else:
            s.new_parameters(show_atoms, probe_radius, gridsp,
                             resolution, level, visible_patches, sharp_boundaries)
            update_color(s, color, transparency)
        surfs.append(s)

    # Close overlapping surfaces.
    if replace:
        other_surfs = set(all_surfs.values()) - set(surfs)
        from chimerax.atomic import concatenate
        surf_atoms = concatenate([s.atoms for s in surfs])
        osurfs = surfaces_overlapping_atoms(other_surfs, surf_atoms)
        if osurfs:
            session.models.close(osurfs)

    # Compute surfaces using multiple threads
    args = [(s,) for s in surfs]
    args.sort(key = lambda s: s[0].atom_count, reverse = True)      # Largest first for load balancing
    from chimerax.core import threadq
    threadq.apply_to_list(_calculate_surface, args, nthread)
#    for s in surfs:
#        s.calculate_surface_geometry()
    # TODO: Any Python error in the threaded call causes a crash when it tries
    #       to write an error message to the log, not in the main thread.

    if not resolution is None and resolution > 0 and level is None:
        gsurfs = [s for s in surfs if hasattr(s, 'gaussian_level')]
        if gsurfs:
            levels = [s.gaussian_level for s in gsurfs]
            min_lev, max_lev = ('%.3f' % min(levels)), ('%.3f' % max(levels))
            level_range = '%s - %s' % (min_lev, max_lev) if max_lev != min_lev else max_lev
            msg = '%d Gaussian surfaces, threshold level %s' % (len(gsurfs), level_range)
            session.logger.info(msg)

    # Add new surfaces to open models list.
    for s, parent in new_surfs:
        session.models.add([s], parent = parent)

    # Make sure replaced surfaces are displayed.
    for s in surfs:
        s.display = True

    # Set automatic updating.
    for s in surfs:
        s.auto_update = update

    return surfs

# -------------------------------------------------------------------------------------
#
def _calculate_surface(surf):
    try:
        surf.calculate_surface_geometry()
    except MemoryError as e:
        from chimerax.core.errors import UserError
        raise UserError(str(e))

# -------------------------------------------------------------------------------------
#
def surface_show_patches(session, objects = None):
    '''
    Show surface patches for atoms of existing surfaces.

    Parameters
    ----------
    objects : Objects
      Show atom patches for existing specified molecular surfaces or for specified atoms.
    '''
    atoms = _surface_atoms(session, objects)

    from chimerax.atomic import molsurf
    surfs = molsurf.show_surface_atom_patches(atoms)
    sset = set(surfs)
    
    # Include molecular surfaces specified by model id.
    sm = _molecular_surfaces(session, objects)
    for s in sm:
        if s not in sset:
            s.show(s.atoms)
            surfs.append(s)

    return surfs

# -------------------------------------------------------------------------------------
#
def surface_hide_patches(session, objects = None):
    '''
    Hide patches of existing surfaces for specified atoms.

    Parameters
    ----------
    objects : Objects
      Hide atom patches for specified molecular surfaces or for specified atoms.
    '''
    atoms = _surface_atoms(session, objects)

    from chimerax.atomic import molsurf
    surfs = molsurf.hide_surface_atom_patches(atoms)
    sset = set(surfs)
    
    # Hide surfaces specified by model id.
    sm = _molecular_surfaces(session, objects)
    for s in sm:
        if s not in sset:
            from chimerax.atomic import Atoms
            s.show(Atoms())
            surfs.append(s)
            
    return surfs

# -------------------------------------------------------------------------------------
#
def surface_close(session, objects = None):
    '''
    Close molecular surfaces.

    Parameters
    ----------
    objects : Objects
      Close specified molecular surfaces and surfaces for specified atoms.
    '''
    surfs = _molecular_surfaces(session, objects)
    from chimerax.atomic.molsurf import close_surfaces
    close_surfaces(surfs)
    if objects:
        close_surfaces(objects.atoms)
        
# -------------------------------------------------------------------------------------
#
def _surface_atoms(session, objects):
    if objects is None:
        from chimerax.atomic import all_atoms
        atoms = all_atoms(session)
    else:
        atoms = objects.atoms
        from chimerax.atomic import MolecularSurface
        satoms = [s.atoms for s in objects.models if isinstance(s, MolecularSurface)]
        if satoms:
            from chimerax.atomic import concatenate, Atoms
            atoms = concatenate([atoms] + satoms, Atoms, remove_duplicates = True)
    return atoms
        
# -------------------------------------------------------------------------------------
#
def _surfaces(session, objects):
    from chimerax.core.models import Surface
    if objects is None:
        sm = session.models.list(type = Surface)
    else:
        sm = [s for s in objects.models if isinstance(s, Surface)]
    return sm
 
# -------------------------------------------------------------------------------------
#
def _molecular_surfaces(session, objects, from_atoms = False):
    from chimerax.atomic import MolecularSurface, surfaces_with_atoms
    if objects is None:
        surfs = session.models.list(type = MolecularSurface)
    else:
        surfs = [s for s in objects.models if isinstance(s, MolecularSurface)]
        if from_atoms:
            surfs.extend(surfaces_with_atoms(objects.atoms))
    return surfs

# -------------------------------------------------------------------------------------
#
def surface_style(session, surfaces, style):
    '''
    Show surface patches for atoms of existing surfaces.

    Parameters
    ----------
    surfaces : Model list
    style : "mesh", "dot" or "solid"
    '''
    if surfaces is None:
        from chimerax.atomic import Structure
        surfaces = [m for m in session.models.list() if not isinstance(m, Structure)]
    from chimerax.map import Volume
    for s in surfaces:
        if isinstance(s, Volume) and s.surface_shown:
            if style == 'dot':
                for d in s.surfaces:
                    d.display_style = d.Dot
            else:
                vstyle = 'surface' if style == 'solid' else 'mesh'
                s.set_display_style(vstyle)
                s.show()
        elif not s.empty_drawing():
            s.display_style = style

# -------------------------------------------------------------------------------------
#
def surface_square_mesh(session, surfaces = None):
    '''
    Show only mesh lines parallel x, y or z axes

    Parameters
    ----------
    surfaces : List of Surface
    '''
    if surfaces is None:
        from chimerax.core.models import Surface
        surfaces = session.models.list(type = Surface)
    from .squaremesh import hide_diagonals
    for s in surfaces:
        hide_diagonals(s)

    return surfaces

# -------------------------------------------------------------------------------------
#
def surface_show_all(session, surfaces = None):
    '''
    Set surface to no masking.

    Parameters
    ----------
    surfaces : List of Surface
    '''
    if surfaces is None:
        from chimerax.core.models import Surface
        surfaces = session.models.list(type = Surface)
    for s in surfaces:
        s.auto_remask_triangles = None
        s.triangle_mask = None
        s.edge_mask = None

    return surfaces

# -------------------------------------------------------------------------------------
#
def surface_cap(session, enable = None, offset = None, subdivision = None, mesh = None):
    '''
    Control whether clipping shows surface caps covering the hole produced by the clip plane.

    Parameters
    ----------
    enable : bool
      Caps are current enabled or disabled for all models, not on a per-model basis.
    offset : float
      Offset of clipping cap from plane in physical units.  Some positive offset is needed or
      the clip plane hides the cap.  Default 0.01.
    subdivision : float
      How small to make the triangles that compose the cap.  Smaller triangles give finer
      appearance with per-vertex coloring.  Default 1.0.  Value of 0 means no subdivision
      and the triangles are long and skinny with one triangle spanning from edge to edge
      of the cap.  Higher values give smaller triangles, for example a value of 2 gives
      triangles twice as small a value of 1.  A value of 1 makes triangles with edges that
      are about the length of the edge lengths on the perimeter of the cap which are usually
      comparable to the size of the triangles of the surface that is being clipped.
    mesh : bool
      Whether mesh style surfaces show caps.  Default False.
    '''
    update = False
    from .settings import settings
    if enable is not None and enable != settings.clipping_surface_caps:
        settings.clipping_surface_caps = enable
        if enable:
            update = True
        else:
            from . import remove_clip_caps
            drawings = session.main_view.drawing.all_drawings()
            remove_clip_caps(drawings)

    if offset is not None:
        settings.clipping_cap_offset = offset
        update = True
        
    if subdivision is not None:
        settings.clipping_cap_subdivision = subdivision
        update = True

    if mesh is not None:
        settings.clipping_cap_on_mesh = mesh
        update = True

    if update:
        clip_planes = session.main_view.clip_planes
        clip_planes.changed = True

    if enable is None and not update:
        onoff = 'on' if settings.clipping_surface_caps else 'off'
        msg = ('Clip caps %s, offset %.3g, subdivision %.3g, mesh %s'
               % (onoff, settings.clipping_cap_offset, settings.clipping_cap_subdivision,
                  settings.clipping_cap_on_mesh))
        session.logger.status(msg, log = True)
        
# -------------------------------------------------------------------------------------
#
def register_command(logger):
    from chimerax.core.commands import CmdDesc, register, ObjectsArg
    from chimerax.core.commands import FloatArg, IntArg, ColorArg, BoolArg, NoArg, create_alias
    from chimerax.core.commands import SurfacesArg, EmptyArg, EnumOf, Or
    from chimerax.atomic import AtomsArg
    surface_desc = CmdDesc(
        optional = [('atoms', AtomsArg)],
        keyword = [('enclose', AtomsArg),
                   ('include', AtomsArg),
                   ('probe_radius', FloatArg),
                   ('grid_spacing', FloatArg),
                   ('resolution', FloatArg),
                   ('level', FloatArg),
                   ('color', ColorArg),
                   ('transparency', FloatArg),
                   ('visible_patches', IntArg),
                   ('sharp_boundaries', BoolArg),
                   ('nthread', IntArg),
                   ('replace', BoolArg),
                   ('update', BoolArg)],
        synopsis = 'create molecular surface')
    register('surface', surface_desc, surface, logger=logger)

    show_desc = CmdDesc(
        optional = [('objects', ObjectsArg)],
        synopsis = 'Show patches of molecular surfaces')
    register('surface showPatches', show_desc, surface_show_patches, logger=logger)

    hide_desc = CmdDesc(
        optional = [('objects', ObjectsArg)],
        synopsis = 'Hide patches of molecular surfaces')
    register('surface hidePatches', hide_desc, surface_hide_patches, logger=logger)
    create_alias('~surface', 'surface hidePatches $*', logger=logger)

    close_desc = CmdDesc(
        optional = [('objects', ObjectsArg)],
        synopsis = 'close molecular surfaces')
    register('surface close', close_desc, surface_close, logger=logger)

    style_desc = CmdDesc(
        required = [('surfaces', Or(SurfacesArg, EmptyArg)),
                    ('style', EnumOf(('mesh', 'dot', 'solid')))],
        synopsis = 'Change surface style to mesh, dot or solid')
    register('surface style', style_desc, surface_style, logger=logger)

    square_desc = CmdDesc(
        optional = [('surfaces', SurfacesArg)],
        synopsis = 'Show only mesh lines parallel x, y or z axes')
    register('surface squaremesh', square_desc, surface_square_mesh, logger=logger)

    showall_desc = CmdDesc(
        optional = [('surfaces', SurfacesArg)],
        synopsis = 'Turn off masking of triangles or edges')
    register('surface showall', showall_desc, surface_show_all, logger=logger)

    cap_desc = CmdDesc(
        optional = [('enable', BoolArg),],
        keyword = [('offset', FloatArg),
                   ('subdivision', FloatArg),
                   ('mesh', BoolArg)],
        synopsis = 'Enable or disable clipping surface caps')
    register('surface cap', cap_desc, surface_cap, logger=logger)

    # Register surface operation subcommands.
    from . import sop
    sop.register_surface_subcommands(logger)

    # Register splitbycolor operation
    from . import splitbycolor
    splitbycolor.register_splitbycolor_command(logger)
    
def check_atoms(atoms, session):
    if atoms is None:
        from chimerax.atomic import all_atoms
        atoms = all_atoms(session)
        if len(atoms) == 0:
            from chimerax.core.errors import UserError
            raise UserError('No atomic models open.')
        atoms.spec = 'all atoms'
    elif len(atoms) == 0:
        msg = 'No atoms specified'
        if hasattr(atoms, 'spec'):
            msg += ' by %s' % atoms.spec
        from chimerax.core.errors import UserError
        raise UserError(msg)
    return atoms
