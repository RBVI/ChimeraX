from ... import cli

# -----------------------------------------------------------------------------
#
def register_fitmap_command():

    from ... import cli, atomspec

    fitmap_desc = cli.CmdDesc(
        required = [
            ('atomsOrMap', atomspec.AtomSpecArg),
        ],
        keyword = [
            ('inMap', atomspec.AtomSpecArg),	# Required, need keyword to avoid two consecutive atom specs.
            ('metric', cli.EnumOf(('overlap', 'correlation', 'cam'))),  # overlap, correlation or cam.
            ('envelope', cli.BoolArg),
            ('resolution', cli.FloatArg),
            ('shift', cli.BoolArg),
            ('rotate', cli.BoolArg),
            ('symmetric', cli.BoolArg),
            ('moveWholeMolecules', cli.BoolArg),
            ('search', cli.IntArg),
            ('placement', cli.EnumOf(('sr', 's', 'r'))),
            ('radius', cli.FloatArg),
            ('clusterAngle', cli.FloatArg),
            ('clusterShift', cli.FloatArg),
            ('asymmetricUnit', cli.BoolArg),
            ('inside', cli.FloatArg),
            ('sequence', cli.IntArg),
            ('maxSteps', cli.IntArg),
            ('gridStepMax', cli.FloatArg),
            ('gridStepMin', cli.FloatArg),
            ('listFits', cli.BoolArg),
            ('eachModel', cli.BoolArg),
        ]
    )
    cli.register('fitmap', fitmap_desc, fitmap)

# -----------------------------------------------------------------------------
#
def fitmap(session, atomsOrMap, inMap = None,
           metric = None, envelope = True, resolution = None,
           shift = True, rotate = True, symmetric = False,
           moveWholeMolecules = True,
           search = 0, placement = 'sr', radius = None,
           clusterAngle = 6, clusterShift = 3,
           asymmetricUnit = True, inside = 0.1, sequence = 0,
           maxSteps = 2000, gridStepMin = 0.01, gridStepMax = 0.5,
           listFits = None, eachModel = False):

  atomsOrMap = atomsOrMap.evaluate(session)

  if inMap is None:
      raise cli.UserError('Must specify "in" keyword, e.g. fit #1 in #2')
  from .. import Volume
  vlist = [v for v in inMap.evaluate(session).models if isinstance(v, Volume)]
  if len(vlist) != 1:
      raise cli.UserError('fitmap second argument must specify one map, got %d' % len(volume))
  volume = vlist[0]

  if listFits is None:
      listFits = (search > 0)

  if eachModel and sequence == 0:
      aomlist = split_selection_by_model(atomsOrMap)
      if not resolution is None:
          aomlist = remove_atoms_with_volumes(aomlist, resolution,
                                              moveWholeMolecules, session)
      flist = []
      for aom in aomlist:
          fits = fitmap(aom, volume, metric, envelope, resolution,
                        shift, rotate, symmetric, moveWholeMolecules, search,
                        placement, radius,
                        clusterAngle, clusterShift, asymmetricUnit,
                        inside, sequence, maxSteps, gridStepMin, gridStepMax,
                        listFits)
          flist.extend(fits)
      return flist

  if metric is None:
      metric = 'correlation' if symmetric else 'overlap'

  if sequence == 0:
      from ...structure import Atoms
      atoms = Atoms(atomsOrMap)
      if len(atoms) == 0:
          v = map_to_fit(atomsOrMap)
      elif resolution is None:
          v = None
      else:
          from . import fitmap as F
          v = F.simulated_map(atoms, resolution, moveWholeMolecules, session)

  if metric in ('correlation', 'cam') and v is None:
      if symmetric:
          raise cli.UserError('Must specify a map resolution for'
                              ' symmetric fitting of an atomic model')
      else:
          raise cli.UserError('Must specify a map resolution when'
                             ' fitting an atomic model using correlation')

  if sequence > 0:
      if search > 0:
          raise cli.UserError('Cannot use "sequence" and "search" options together.')
      if symmetric:
          raise cli.UserError('Cannot use "sequence" and "symmetric" options together.')
      flist = fit_sequence(atomsOrMap, volume, session, metric, envelope, resolution,
                           shift, rotate, moveWholeMolecules, sequence,
                           maxSteps, gridStepMin, gridStepMax)
  elif search == 0:
      log = session.logger
      if v is None:
          f = fit_atoms_in_map(atoms, volume, shift, rotate, moveWholeMolecules,
                               maxSteps, gridStepMin, gridStepMax, log)
          flist = [f]
      else:
          if symmetric:
              if not metric in ('correlation', 'cam'):
                  raise cli.UserError('Only "correlation" and "cam" metrics are'
                                     ' supported with symmetric fitting')
              if len(volume.data.symmetries) == 0:
                  raise cli.UserError('Volume %s has not symmetry assigned'
                                     % volume.name)
              f = fit_map_in_symmetric_map(v, volume, metric, envelope,
                                           shift, rotate, moveWholeMolecules,
                                           atoms,
                                           maxSteps, gridStepMin, gridStepMax, log)
              flist = [f]
          else:
              f = fit_map_in_map(v, volume, metric, envelope,
                                 shift, rotate, moveWholeMolecules, atoms,
                                 maxSteps, gridStepMin, gridStepMax, log)
              flist = [f]
  else:
      if symmetric:
          raise cli.UserError('Symmetric fitting not available with fit search')
      flist = fit_search(atoms, v, volume, metric, envelope, shift, rotate,
                         moveWholeMolecules, search, placement, radius,
                         clusterAngle, clusterShift, asymmetricUnit, inside,
                         maxSteps, gridStepMin, gridStepMax, session.logger)

  if listFits:
      from . import fitlist
      d = fitlist.show_fit_list_dialog(session)
      d.add_fits(flist)
      if search > 0 and len(flist) > 0:
          d.select_fit(flist[0])

  return flist

# -----------------------------------------------------------------------------
#
def split_selection_by_model(sel):

    ma = {}
    for a in sel.atoms():
        m = a.molecule
        if m in ma:
            ma[m].append(a)
        else:
            ma[m] = [a]
    from .. import Volume
    from chimera.selection import ItemizedSelection
    vsel = [ItemizedSelection(v) for v in sel.models() if isinstance(v, Volume)]
    msel = [ItemizedSelection(ma[m]) for m in sel.models() if m in ma]
    return vsel + msel

# -----------------------------------------------------------------------------
# When fitting each model exclude atoms where a corresponding simulated map
# is also specified.
#
def remove_atoms_with_volumes(aomlist, res, mwm, session):

    faomlist = []
    models = set(sum([aom.models() for aom in aomlist],[]))
    from .fitmap import find_simulated_map
    for aom in aomlist:
        atoms = aom.atoms()
        if (len(atoms) == 0 or
            not find_simulated_map(atoms, res, mwm, session) in models):
            faomlist.append(aom)
    return faomlist

# -----------------------------------------------------------------------------
#
def fit_atoms_in_map(atoms, volume, shift, rotate, moveWholeMolecules,
                     maxSteps, gridStepMin, gridStepMax, log = None):

    from . import fitmap as F
    stats = F.move_atoms_to_maximum(atoms, volume,
                                    maxSteps, gridStepMin, gridStepMax, 
                                    shift, rotate, moveWholeMolecules,
                                    request_stop_cb = report_status(log))
    mols = atoms.molecules
    if log and stats:
        log.info(F.atom_fit_message(mols, volume, stats))
        if moveWholeMolecules:
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
                   shift, rotate, moveWholeMolecules, mapAtoms,
                   maxSteps, gridStepMin, gridStepMax, log = None):

    me = fitting_metric(metric)
    points, point_weights = map_fitting_points(v, envelope)
    symmetries = []

    from . import fitmap as F
    move_tf, stats = F.motion_to_maximum(points, point_weights, volume,
                                         maxSteps, gridStepMin, gridStepMax,
                                         shift, rotate, me, symmetries,
                                         report_status(log))
    from . import move
    move.move_models_and_atoms(move_tf, [v], mapAtoms, moveWholeMolecules,
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
                             shift, rotate, moveWholeMolecules, mapAtoms,
                             maxSteps, gridStepMin, gridStepMax, log = None):

    from . import fitmap as F
    me = fitting_metric(metric)
    points, point_weights = map_fitting_points(volume, envelope,
                                               local_coords = True)
    refpt = F.volume_center_point(v, volume.openState.xform)
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
                                          maxSteps, gridStepMin, gridStepMax,
                                          shift, rotate, me, refpt, symmetries,
                                          stop_cb)
    finally:
        task.finished()

    if stats is None:
        return          # Fit cancelled

    ctf = volume.position.inverse()
    vtf = ctf.inverse() * move_tf.inverse() * ctf
    from . import move
    move.move_models_and_atoms(vtf, [v], mapAtoms, moveWholeMolecules, volume)

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
               moveWholeMolecules, search, placement, radius,
               clusterAngle, clusterShift, asymmetricUnit, inside,
               maxSteps, gridStepMin, gridStepMax, log = None):
    
    # TODO: Handle case where not moving whole molecules.

    me = fitting_metric(metric)
    if v is None:
        points = atoms.coordinates()
        volume.position.inverse().move(points)
        point_weights = None
    else:
        points, point_weights = map_fitting_points(v, envelope)
        import Matrix
        Matrix.xform_points(points, volume.openState.xform.inverse())
    from . import search as FS
    rotations = 'r' in placement
    shifts = 's' in placement
    mlist = list(atoms.molecules)
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
            radius, clusterAngle, clusterShift, asymmetricUnit, inside,
            me, shift, rotate, maxSteps, gridStepMin, gridStepMax, stop_cb)
#    finally:
#        task.finished()

    if log:
        report_fit_search_results(flist, search, outside, inside, log)
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
def report_fit_search_results(flist, search, outside, inside, log):

    log.info('Found %d unique fits from %d random placements ' %
             (len(flist), search) +
             'having fraction of points inside contour >= %.3f (%d of %d).\n'
             % (inside, search-outside,  search))

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
def fit_sequence(atomsOrMap, volume, session, metric, envelope, resolution,
                 shift, rotate, moveWholeMolecules, sequence,
                 maxSteps, gridStepMin, gridStepMax):

    if resolution is None:
        from .. import Volume
        vlist = [v for v in atomsOrMap.models if isinstance(v, Volume)]
        if len(vlist) < 2:
            if len(atomsOrMap.atoms) > 0:
                raise cli.UserError('Fit sequence requires "resolution"'
                                   ' argument to fit atomic models')
            else:
                raise cli.UserError('Fit sequence requires 2 or more maps'
                                   ' to place')
    else:
        from ...structure import StructureModel
        mlist = [m for m in atomsOrMap.models if isinstance(m, StructureModel)]
        if len(mlist) < 2:
            raise cli.UserError('Fit sequence requires 2 or more molecules')
        if not moveWholeMolecules:
            raise cli.UserError('Fit sequence does not support'
                               ' moving partial molecules')
            # TODO: Handle case where not moving whole molecules.
        import fitmap as F
        vlist = [F.simulated_map(m.atoms, resolution, moveWholeMolecules, session)
                 for m in mlist]

    me = fitting_metric(metric)

    import sequence as S
    from chimera import tasks
    task = tasks.Task("Fit sequence", modal=True)
    def stop_cb(msg, task = task):
        return request_stop_cb(msg, task)
    flist = []
    try:
        flist = S.fit_sequence(vlist, volume, sequence,
                               envelope, me, shift, rotate,
                               maxSteps, gridStepMin, gridStepMax, stop_cb)
    finally:
        task.finished()

    return flist

# -----------------------------------------------------------------------------
#
def map_to_fit(selection):

  from .. import Volume
  vlist = [m for m in selection.models() if isinstance(m, Volume)]
  if len(vlist) == 0:
    raise cli.UserError('No atoms or maps for %s' % selection.oslStr)
  elif len(vlist) > 1:
    raise cli.UserError('Multiple maps for %s' % selection.oslStr)
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
    raise cli.UserError('metric "%s" must be "overlap" or "correlation"' % metric)
  return me

# -----------------------------------------------------------------------------
# Returns points in global coordinates.
#
def map_fitting_points(v, envelope, local_coords = False):

    from ...geometry.place import identity
    point_to_scene_transform = None if local_coords else identity()
    from . import fitmap as F
    try:
        points, point_weights = F.map_points_and_weights(v, envelope,
                                                         point_to_scene_transform)
    except (MemoryError, ValueError) as e:
        raise cli.UserError('Out of memory, too many points in %s' % v.name)

    if len(points) == 0:
        msg = ('No grid points above map threshold.' if envelope
               else 'Map has no non-zero values.')
        raise cli.UserError(msg)

    return points, point_weights

# -----------------------------------------------------------------------------
#
def report_status(log):
    return None if log is None else lambda message, log=log: log.status(message)

