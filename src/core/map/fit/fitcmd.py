from ...errors import UserError

# -----------------------------------------------------------------------------
#
def fitmap(session, atoms_or_map, in_map = None, subtract_maps = None,
           metric = None, envelope = True, resolution = None,
           shift = True, rotate = True, symmetric = False,
           move_whole_molecules = True,
           search = 0, placement = 'sr', radius = None,
           cluster_angle = 6, cluster_shift = 3,
           asymmetric_unit = True, level_inside = 0.1, sequence = 0,
           max_steps = 2000, grid_step_min = 0.01, grid_step_max = 0.5,
           list_fits = None, each_model = False):
    '''
    Fit an atomic model or a map in a map using a rigid rotation and translation
    by locally optimizing correlation.  There are four modes: 1) fit all models into map
    as a single rigid group, 2) fit each model into the map separately, 3) fit each model
    separately while subtracting others (sequence mode), or 4) use random starting positions
    when fitting (search mode).

    Parameters
    ----------
    atoms_or_map
      Atoms or map that will be moved.
    in_map : Volume
      Target density map to fit into.
    subtract_maps : Atoms
      Subtract map for these atoms from the target map before fitting.

    ----------------------------------------------------------------------------
    Mode
    ----------------------------------------------------------------------------
    each_model : bool
      Whether to fit each model independently or all as one rigid group.
    sequence : integer
      Fit each model in sequence subtracting other models first for this number of specified fits.
    search : integer
      Fit using N randomized initial placements and cluster similar results.

    ----------------------------------------------------------------------------
    Fitting settings
    ----------------------------------------------------------------------------
    metric : 'overlap', 'correlation', or 'cam'
      Optimization function to use.  Overlap is pointwise sum.  Cam is correlation
      about mean (ie. mean value is subtracted from maps before computing correlation).
    envelope : bool
      Whether to consider fit only within lowest displayed contour level if moving map.
    resolution : float
      Resolution for making simulated maps from atomic models.  Required when correlation
      or cam metric is used and atomic models are being fit.
    shift : bool
      Allow translation when fitting.
    rotate : bool
      Allow rotation when fitting.
    symmetric : bool
      Whether to use symmetry of the target map to take account of clashes between
      copies of the fit models.
    max_steps: integer
      Maximum number of gradient ascent steps to take
    grid_step_max : float
      Maximum motion during a fitting step in grid index units.
    grid_step_min : float
      The fit is considered converged when Motion less than this value (grid index units).

    ----------------------------------------------------------------------------
    Search options
    ----------------------------------------------------------------------------
    placement : 'sr', 's', or 'r'
      Whether random placements should include shift and rotation
    radius : float
      Limits the random placements to within this distance of the starting position.
    cluster_angle : float
      Rotational difference for a fit to form a new cluster.
    cluster_shift : float
      Shift difference for a fit to form a new cluster.
    asymmetric_unit : bool
      List only one symmetrically equivalent fit position if the target map has symmetry.
    level_inside : float
       Fraction of fit atoms or map that must be inside the target map contour level
       in order to keep the fit.

    ----------------------------------------------------------------------------
    Output options
    ----------------------------------------------------------------------------
    move_whole_molecules : bool
      Move entire molecules, or only the atoms that were specified.
    list_fits : bool
      Show the fits in a dialog.
    '''
    spec = str(atoms_or_map)
    atoms_or_map = atoms_or_map.evaluate(session)
    atoms_or_map.spec = spec
    volume = in_map
    if metric is None:
        metric = 'correlation' if symmetric else 'overlap'
    mwm = move_whole_molecules
    if subtract_maps:
        smaps = subtraction_maps(subtract_maps.evaluate(session), resolution, session)
        if sequence == 0:
            sequence = 1

    check_fit_options(atoms_or_map, volume, metric, resolution,
                      symmetric, mwm, search, sequence)

    flist = []
    log = session.logger
    amlist = atoms_and_map(atoms_or_map, resolution, mwm, sequence, each_model, session)
    for atoms, v in amlist:
        if sequence > 0 or subtract_maps:
            fits = fit_sequence(v, volume, smaps, metric, envelope, resolution,
                                shift, rotate, mwm, sequence,
                                max_steps, grid_step_min, grid_step_max, log)
        elif search:
            fits = fit_search(atoms, v, volume, metric, envelope, shift, rotate,
                              mwm, search, placement, radius,
                              cluster_angle, cluster_shift, asymmetric_unit, level_inside,
                              max_steps, grid_step_min, grid_step_max, log)
        elif v is None:
            fits = [fit_atoms_in_map(atoms, volume, shift, rotate, mwm,
                                     max_steps, grid_step_min, grid_step_max, log)]
        elif symmetric:
            fits = [fit_map_in_symmetric_map(v, volume, metric, envelope,
                                             shift, rotate, mwm,
                                             atoms, max_steps, grid_step_min, grid_step_max, log)]
        else:
            fits = [fit_map_in_map(v, volume, metric, envelope,
                                   shift, rotate, mwm, atoms,
                                   max_steps, grid_step_min, grid_step_max, log)]

        if list_fits:
            show_first = search > 0
            list_fits(fits, show_first, session)
        flist.extend(fits)

    return flist

# -----------------------------------------------------------------------------
#
def check_fit_options(atoms_or_map, volume, metric, resolution,
                      symmetric, move_whole_molecules, search, sequence):
    if volume is None:
        raise UserError('Must specify "in" keyword, e.g. fit #1 in #2')
    if sequence > 0 and search > 0:
        raise UserError('Cannot use "sequence" and "search" options together.')
    if sequence > 0 and symmetric:
        raise UserError('Cannot use "sequence" and "symmetric" options together.')
    if search and symmetric:
        raise UserError('Symmetric fitting not available with fit search')
    if symmetric:
      if not metric in ('correlation', 'cam', None):
          raise UserError('Only "correlation" and "cam" metrics are'
                          ' supported with symmetric fitting')
      if len(volume.data.symmetries) == 0:
          raise UserError('Volume %s has not symmetry assigned' % volume.name)
    if len(atoms_or_map.atoms) > 0 and resolution is None:
        if symmetric:
            raise UserError('Must specify a map resolution for'
                            ' symmetric fitting of an atomic model')
        elif metric in ('correlation', 'cam'):
            raise UserError('Must specify a map resolution when'
                            ' fitting an atomic model using correlation')

    if sequence > 0 and not move_whole_molecules:
        raise UserError('Fit sequence does not support'
                        ' moving partial molecules')

# -----------------------------------------------------------------------------
#
def atoms_and_map(atoms_or_map, resolution, move_whole_molecules, sequence, each_model, session):

    if each_model and sequence == 0:
        return split_by_model(atoms_or_map, resolution, session)

    if sequence > 0:
        if resolution is None:
            from .. import Volume
            vlist = [v for v in atoms_or_map.models if isinstance(v, Volume)]
            if len(vlist) < 2:
                if len(atoms_or_map.atoms) > 0:
                    raise UserError('Fit sequence requires "resolution"'
                                    ' argument to fit atomic models')
                else:
                    raise UserError('Fit sequence requires 2 or more maps to place')
        else:
            from ...structure import AtomicStructure
            mlist = [m for m in atoms_or_map.models if isinstance(m, AtomicStructure)]
            if len(mlist) == 0:
                raise UserError('No molecules specified for fitting')
            from . import fitmap as F
            vlist = [F.simulated_map(m.atoms, resolution, session) for m in mlist]
        from ...molecule import Atoms
        return [(Atoms(), vlist)]

    atoms = atoms_or_map.atoms
    if len(atoms) == 0:
        v = map_to_fit(atoms_or_map)
    elif resolution is None:
        v = None
    else:
        from . import fitmap as F
        v = F.simulated_map(atoms, resolution, session)
    return [(atoms, v)]

# -----------------------------------------------------------------------------
#
def subtraction_maps(spec, resolution, session):
    vlist = []
    atoms = spec.atoms
    if len(atoms) > 0:
        if resolution is None:
            raise UserError('Require resolution keyword for atomic models used '
                            'in subtract maps option')
        for m, matoms in atoms.by_structure:
            from .fitmap import simulated_map
            vlist.append(simulated_map(matoms, resolution, session))
    from .. import Volume
    vlist.extend([v for v in spec.models if isinstance(v, Volume)])
    return vlist
    
# -----------------------------------------------------------------------------
#
def split_by_model(sel, resolution, session):

    aom = [(atoms, None) for m,atoms in sel.atoms.by_structure]
    from .. import Volume
    aom.extend([(None, v) for v in sel.models if isinstance(v, Volume)])
    if not resolution is None:
        aom = remove_atoms_with_volumes(aom, resolution, session)
    return aom

# -----------------------------------------------------------------------------
# When fitting each model exclude atoms where a corresponding simulated map
# is also specified.
#
def remove_atoms_with_volumes(aom, res, session):

    maps = set(v for v,a in aom)
    from .fitmap import find_simulated_map
    faom = [(atoms,v) for atoms,v in aom
            if atoms is None or not find_simulated_map(atoms, res, session) in maps]
    return faom

# -----------------------------------------------------------------------------
#
def list_fits(flist, show, session):
    from . import fitlist
    d = fitlist.show_fit_list_dialog(session)
    d.add_fits(flist)
    if show and len(flist) > 0:
        d.select_fit(flist[0])

# -----------------------------------------------------------------------------
#
def fit_atoms_in_map(atoms, volume, shift, rotate, move_whole_molecules,
                     max_steps, grid_step_min, grid_step_max, log = None):

    from . import fitmap as F
    stats = F.move_atoms_to_maximum(atoms, volume,
                                    max_steps, grid_step_min, grid_step_max, 
                                    shift, rotate, move_whole_molecules,
                                    request_stop_cb = report_status(log))
    mols = atoms.unique_structures
    if log and stats:
        log.info(F.atom_fit_message(mols, volume, stats))
        if move_whole_molecules:
            for m in mols:
                log.info(F.transformation_matrix_message(m, volume))
        log.status('%d steps, shift %.3g, rotation %.3g degrees, average map value %.4g'
                   % (stats['steps'], stats['shift'], stats['angle'], stats['average map value']))

    from .search import Fit
    fit = Fit(mols, None, volume, stats)
    return fit

# -----------------------------------------------------------------------------
#
def fit_map_in_map(v, volume, metric, envelope,
                   shift, rotate, move_whole_molecules, map_atoms,
                   max_steps, grid_step_min, grid_step_max, log = None):

    me = fitting_metric(metric)
    points, point_weights = map_fitting_points(v, envelope)
    symmetries = []

    from . import fitmap as F
    move_tf, stats = F.motion_to_maximum(points, point_weights, volume,
                                         max_steps, grid_step_min, grid_step_max,
                                         shift, rotate, me, symmetries,
                                         report_status(log))
    from . import move
    move.move_models_and_atoms(move_tf, [v], map_atoms, move_whole_molecules,
                               volume)

    if log:
        log.info(F.map_fit_message(v, volume, stats))
        log.info(F.transformation_matrix_message(v, volume))
        cort = me if me == 'correlation about mean' else 'correlation'
        log.status('%d steps, shift %.3g, rotation %.3g degrees, %s %.4f'
                   % (stats['steps'], stats['shift'], stats['angle'], cort, stats[cort]))

    from .search import Fit
    fit = Fit([v], None, volume, stats)
    return fit

# -----------------------------------------------------------------------------
#
def fit_map_in_symmetric_map(v, volume, metric, envelope,
                             shift, rotate, move_whole_molecules, map_atoms,
                             max_steps, grid_step_min, grid_step_max, log = None):

    from . import fitmap as F
    me = fitting_metric(metric)
    points, point_weights = map_fitting_points(volume, envelope,
                                               local_coords = True)
    refpt = F.volume_center_point(v, volume.place)
    symmetries = volume.data.symmetries
    indices = F.asymmetric_unit_points(points, refpt, symmetries)
    apoints = points[indices]
    apoint_weights = point_weights[indices]

    data_array, xyz_to_ijk_transform = \
      v.matrix_and_transform(volume.position, subregion = None, step = 1)

    from chimera import tasks, CancelOperation
    task = tasks.Task("Symmetric fit", modal=True)
    def stop_cb(msg, task = task):
        return request_stop_cb(msg, task)
    stats = None
    try:
        move_tf, stats = F.locate_maximum(apoints, apoint_weights,
                                          data_array, xyz_to_ijk_transform,
                                          max_steps, grid_step_min, grid_step_max,
                                          shift, rotate, me, refpt, symmetries,
                                          stop_cb)
    finally:
        task.finished()

    if stats is None:
        return          # Fit cancelled

    ctf = volume.position.inverse()
    vtf = ctf.inverse() * move_tf.inverse() * ctf
    from . import move
    move.move_models_and_atoms(vtf, [v], map_atoms, move_whole_molecules, volume)

    if log:
        log.info(F.map_fit_message(v, volume, stats))
        log.info(F.transformation_matrix_message(v, volume))
        cort = me if me == 'correlation about mean' else 'correlation'
        log.status('%d steps, shift %.3g, rotation %.3g degrees, %s %.4f'
                   % (stats['steps'], stats['shift'], stats['angle'], cort, stats[cort]))

    from .search import Fit
    fit = Fit([v], None, volume, stats)
    return fit

# -----------------------------------------------------------------------------
#
def fit_search(atoms, v, volume, metric, envelope, shift, rotate,
               move_whole_molecules, search, placement, radius,
               cluster_angle, cluster_shift, asymmetric_unit, level_inside,
               max_steps, grid_step_min, grid_step_max, log = None):
    
    # TODO: Handle case where not moving whole molecules.

    me = fitting_metric(metric)
    if v is None:
        points = atoms.scene_coords
        volume.position.inverse().move(points)
        point_weights = None
    else:
        points, point_weights = map_fitting_points(v, envelope)
        points = volume.place.inverse()*points
    from . import search as FS
    rotations = 'r' in placement
    shifts = 's' in placement
    mlist = list(atoms.unique_structures)
    if v:
        mlist.append(v)

#    from chimera import tasks
#    task = tasks.Task("Fit search", modal=True)
    task = None
    def stop_cb(msg, task = task):
        return request_stop_cb(msg, task)
#    flist = []
#    try:
    flist, outside = FS.fit_search(
            mlist, points, point_weights, volume, search, rotations, shifts,
            radius, cluster_angle, cluster_shift, asymmetric_unit, level_inside,
            me, shift, rotate, max_steps, grid_step_min, grid_step_max, stop_cb)
#    finally:
#        task.finished()

    if log:
        report_fit_search_results(flist, search, outside, level_inside, log)
    return flist

# -----------------------------------------------------------------------------
#
def request_stop_cb(message, task):

#    from chimera import CancelOperation
#    try:
#        task.updateStatus(message)
#    except CancelOperation:
#        return True
    return False

# -----------------------------------------------------------------------------
#
def report_fit_search_results(flist, search, outside, level_inside, log):

    log.info('Found %d unique fits from %d random placements ' %
             (len(flist), search) +
             'having fraction of points inside contour >= %.3f (%d of %d).\n'
             % (level_inside, search-outside,  search))

    if len(flist) == 0:
        return

    f0 = flist[0]
    v = f0.fit_map()
    sattr = 'correlation' if v else 'average_map_value'
    scores = ', '.join(['%.4g (%d)' % (getattr(f,sattr)(),f.hits())
                        for f in flist])
    sname = 'Correlations' if v else 'Average map values'
    log.info('%s and times found:\n\t%s\n' % (sname, scores))
    log.info('Best fit found:\n%s' % f0.fit_message())

# -----------------------------------------------------------------------------
#
def fit_sequence(vlist, volume, subtract_maps = [], metric = 'overlap', envelope = True,
                 resolution = None, shift = True, rotate = True, move_whole_molecules = True,
                 sequence = 1, max_steps = 2000, grid_step_min = 0.01, grid_step_max = 0.5,
                 log = None):

    me = fitting_metric(metric)

    from . import sequence as S
#    from chimera import tasks
#    task = tasks.Task("Fit sequence", modal=True)
#    def stop_cb(msg, task = task):
#        return request_stop_cb(msg, task)
    stop_cb = None
    flist = []
#    try:
    flist = S.fit_sequence(vlist, volume, sequence, subtract_maps, envelope, me, shift, rotate,
                           max_steps, grid_step_min, grid_step_max, stop_cb, log)
#    finally:
#        task.finished()

    # Align molecules to their corresponding maps. 
    # TODO: Handle case where not moving whole molecules.
    for v in vlist: 
        if hasattr(v, 'atoms'):
            for m in v.atoms.unique_structures:
                m.position = v.position
 
    return flist

# -----------------------------------------------------------------------------
#
def map_to_fit(selection):

  from .. import Volume
  vlist = [m for m in selection.models if isinstance(m, Volume)]
  if len(vlist) == 0:
    raise UserError('No atoms or maps for %s' % selection.spec)
  elif len(vlist) > 1:
    raise UserError('Multiple maps for %s' % selection.spec)
  v = vlist[0]
  return v

# -----------------------------------------------------------------------------
#
def fitting_metric(metric):

  if 'overlap'.startswith(metric):
    me = 'sum product'
  elif 'correlation'.startswith(metric):
    me = 'correlation'
  elif 'cam'.startswith(metric):
    me = 'correlation about mean'
  else:
    raise UserError('metric "%s" must be "overlap" or "correlation" or "cam"' % metric)
  return me

# -----------------------------------------------------------------------------
# Returns points in global coordinates.
#
def map_fitting_points(v, envelope, local_coords = False):

    from ...geometry import identity
    point_to_scene_transform = None if local_coords else identity()
    from . import fitmap as F
    try:
        points, point_weights = F.map_points_and_weights(v, envelope,
                                                         point_to_scene_transform)
    except (MemoryError, ValueError) as e:
        raise UserError('Out of memory, too many points in %s' % v.name)

    if len(points) == 0:
        msg = ('No grid points above map threshold.' if envelope
               else 'Map has no non-zero values.')
        raise UserError(msg)

    return points, point_weights

# -----------------------------------------------------------------------------
#
def report_status(log):
    return None if log is None else lambda message, log=log: log.status(message)

# -----------------------------------------------------------------------------
#
def register_fitmap_command():

    from ...commands import CmdDesc, register, BoolArg, IntArg, FloatArg, EnumOf, AtomSpecArg
    from ..mapargs import MapArg

    fitmap_desc = CmdDesc(
        required = [
            ('atoms_or_map', AtomSpecArg),
        ],
        keyword = [
            ('in_map', MapArg),	# Require keyword to avoid two consecutive atom specs.
            ('subtract_maps', AtomSpecArg),

# Four modes, default is single fit mode (no option)
            ('each_model', BoolArg),
            ('sequence', IntArg),
            ('search', IntArg),

# Fitting settings
            ('metric', EnumOf(('overlap', 'correlation', 'cam'))),  # overlap, correlation or cam.
            ('envelope', BoolArg),
            ('resolution', FloatArg),
            ('shift', BoolArg),
            ('rotate', BoolArg),
            ('symmetric', BoolArg),
            ('max_steps', IntArg),
            ('grid_step_max', FloatArg),
            ('grid_step_min', FloatArg),

# Search options
            ('placement', EnumOf(('sr', 's', 'r'))),
            ('radius', FloatArg),
            ('cluster_angle', FloatArg),
            ('cluster_shift', FloatArg),
            ('asymmetric_unit', BoolArg),
            ('level_inside', FloatArg),            # fraction of point in contour

# Output options
            ('move_whole_molecules', BoolArg),
            ('list_fits', BoolArg),
        ]
    )
    register('fitmap', fitmap_desc, fitmap)
