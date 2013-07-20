from ..commands import CommandError

# -----------------------------------------------------------------------------
#
def fitmap_command(cmdname, args):

    from ..commands import parse_arguments
    from ..commands import specifier_arg, volume_arg, int_arg, float_arg, bool_arg
    from ..commands import string_arg

    req_args = (('atomsOrMap', specifier_arg),
                ('volume', volume_arg),
                )
    opt_args = ()
    kw_args = (('metric', string_arg),  # overlap, correlation or cam.
               ('envelope', bool_arg),
               ('resolution', float_arg),
               ('shift', bool_arg),
               ('rotate', bool_arg),
               ('symmetric', bool_arg),
               ('moveWholeMolecules', bool_arg),
               ('search', int_arg),
               ('placement', string_arg),
               ('radius', float_arg),
               ('clusterAngle', float_arg),
               ('clusterShift', float_arg),
               ('asymmetricUnit', bool_arg),
               ('inside', float_arg),
               ('sequence', int_arg),
               ('maxSteps', int_arg),
               ('gridStepMax', float_arg),
               ('gridStepMin', float_arg),
               ('listFits', bool_arg),
               ('eachModel', bool_arg),
               )
    kw = parse_arguments(cmdname, args, req_args, opt_args, kw_args)
    fitmap(**kw)

# -----------------------------------------------------------------------------
#
def fitmap(atomsOrMap, volume,
           metric = None, envelope = True, resolution = None,
           shift = True, rotate = True, symmetric = False,
           moveWholeMolecules = True,
           search = 0, placement = 'sr', radius = None,
           clusterAngle = 6, clusterShift = 3,
           asymmetricUnit = True, inside = 0.1, sequence = 0,
           maxSteps = 2000, gridStepMin = 0.01, gridStepMax = 0.5,
           listFits = None, eachModel = False):

  if listFits is None:
      listFits = (search > 0)

  if eachModel and sequence == 0:
      aomlist = split_selection_by_model(atomsOrMap)
      if not resolution is None:
          aomlist = remove_atoms_with_volumes(aomlist, resolution,
                                              moveWholeMolecules)
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
      atoms = atomsOrMap.atom_set()
      if atoms.count() == 0:
          v = map_to_fit(atomsOrMap)
      elif resolution is None:
          v = None
      else:
          from . import fitmap as F
          v = F.simulated_map(atoms, resolution, moveWholeMolecules)

  if metric in ('correlation', 'cam') and v is None:
      if symmetric:
          raise CommandError('Must specify a map resolution for'
                             ' symmetric fitting of an atomic model')
      else:
          raise CommandError('Must specify a map resolution when'
                             ' fitting an atomic model using correlation')

  if sequence > 0:
      if search > 0:
          raise CommandError('Cannot use "sequence" and "search" options together.')
      if symmetric:
          raise CommandError('Cannot use "sequence" and "symmetric" options together.')
      flist = fit_sequence(atomsOrMap, volume, metric, envelope, resolution,
                           shift, rotate, moveWholeMolecules, sequence,
                           maxSteps, gridStepMin, gridStepMax)
  elif search == 0:
      if v is None:
          f = fit_atoms_in_map(atoms, volume, shift, rotate, moveWholeMolecules,
                               maxSteps, gridStepMin, gridStepMax)
          flist = [f]
      else:
          if symmetric:
              if not metric in ('correlation', 'cam'):
                  raise CommandError('Only "correlation" and "cam" metrics are'
                                     ' supported with symmetric fitting')
              if len(volume.data.symmetries) == 0:
                  raise CommandError('Volume %s has not symmetry assigned'
                                     % volume.name)
              f = fit_map_in_symmetric_map(v, volume, metric, envelope,
                                           shift, rotate, moveWholeMolecules,
                                           atoms,
                                           maxSteps, gridStepMin, gridStepMax)
              flist = [f]
          else:
              f = fit_map_in_map(v, volume, metric, envelope,
                                 shift, rotate, moveWholeMolecules, atoms,
                                 maxSteps, gridStepMin, gridStepMax)
              flist = [f]
  else:
      if symmetric:
          raise CommandError('Symmetric fitting not available with fit search')
      flist = fit_search(atoms, v, volume, metric, envelope, shift, rotate,
                         moveWholeMolecules, search, placement, radius,
                         clusterAngle, clusterShift, asymmetricUnit, inside,
                         maxSteps, gridStepMin, gridStepMax)

  if listFits:
      from . import fitlist
      d = fitlist.show_fit_list_dialog()
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
    from .VolumeViewer import Volume
    from chimera.selection import ItemizedSelection
    vsel = [ItemizedSelection(v) for v in sel.models() if isinstance(v, Volume)]
    msel = [ItemizedSelection(ma[m]) for m in sel.models() if m in ma]
    return vsel + msel

# -----------------------------------------------------------------------------
# When fitting each model exclude atoms where a corresponding simulated map
# is also specified.
#
def remove_atoms_with_volumes(aomlist, res, mwm):

    faomlist = []
    models = set(sum([aom.models() for aom in aomlist],[]))
    from .fitmap import find_simulated_map
    for aom in aomlist:
        atoms = aom.atoms()
        if (len(atoms) == 0 or
            not find_simulated_map(atoms, res, mwm) in models):
            faomlist.append(aom)
    return faomlist

# -----------------------------------------------------------------------------
#
def fit_atoms_in_map(atoms, volume, shift, rotate, moveWholeMolecules,
                     maxSteps, gridStepMin, gridStepMax):

    from . import fitmap as F
    stats = F.move_atoms_to_maximum(atoms, volume,
                                    maxSteps, gridStepMin, gridStepMax, 
                                    shift, rotate, moveWholeMolecules,
                                    request_stop_cb = report_status)
    mols = atoms.molecules()
    if stats:
        from ..gui import show_info, show_status
        show_info(F.atom_fit_message(atoms.molecules(), volume, stats))
        if moveWholeMolecules:
            for m in mols:
                show_info(F.transformation_matrix_message(m, volume))
        ave = stats['average map value']
        show_status(', average map value %.4g' % ave, append = True)

    from .search import Fit
    fit = Fit(mols, None, volume, stats)
    return fit

# -----------------------------------------------------------------------------
#
def fit_map_in_map(v, volume, metric, envelope,
                   shift, rotate, moveWholeMolecules, mapAtoms,
                   maxSteps, gridStepMin, gridStepMax):

    me = fitting_metric(metric)
    points, point_weights = map_fitting_points(v, envelope)
    symmetries = []
    from . import fitmap as F
    move_tf, stats = F.motion_to_maximum(points, point_weights, volume,
                                         maxSteps, gridStepMin, gridStepMax,
                                         shift, rotate, me, symmetries,
                                         report_status)
    from . import move
    move.move_models_and_atoms(move_tf, [v], mapAtoms, moveWholeMolecules,
                               volume)

    from ..gui import show_info, show_status
    show_info(F.map_fit_message(v, volume, stats))
    show_info(F.transformation_matrix_message(v, volume))
    cort = me if me == 'correlation about mean' else 'correlation'
    show_status(', %s %.4f' % (cort, stats[cort]), append = True)

    from .search import Fit
    fit = Fit([v], None, volume, stats)
    return fit

# -----------------------------------------------------------------------------
#
def fit_map_in_symmetric_map(v, volume, metric, envelope,
                             shift, rotate, moveWholeMolecules, mapAtoms,
                             maxSteps, gridStepMin, gridStepMax):

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
      v.matrix_and_transform(volume.place, subregion = None, step = 1)

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

    ctf = volume.place.inverse()
    vtf = ctf.inverse() * move_tf.inverse() * ctf
    from . import move
    move.move_models_and_atoms(vtf, [v], mapAtoms, moveWholeMolecules, volume)

    from ..gui import show_info, show_status
    show_info(F.map_fit_message(v, volume, stats))
    show_info(F.transformation_matrix_message(v, volume))
    cort = me if me == 'correlation about mean' else 'correlation'
    show_status(', %s %.4f' % (cort, stats[cort]), append = True)

    from .search import Fit
    fit = Fit([v], None, volume, stats)
    return fit

# -----------------------------------------------------------------------------
#
def fit_search(atoms, v, volume, metric, envelope, shift, rotate,
               moveWholeMolecules, search, placement, radius,
               clusterAngle, clusterShift, asymmetricUnit, inside,
               maxSteps, gridStepMin, gridStepMax):
    
    # TODO: Handle case where not moving whole molecules.

    me = fitting_metric(metric)
    if v is None:
        points = atoms.coordinates()
        volume.place.inverse().move(points)
        point_weights = None
    else:
        points, point_weights = map_fitting_points(v, envelope)
        import Matrix
        Matrix.xform_points(points, volume.openState.xform.inverse())
    from . import search as FS
    rotations = 'r' in placement
    shifts = 's' in placement
    mlist = atoms.molecules()
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

    report_fit_search_results(flist, search, outside, inside)
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
def report_fit_search_results(flist, search, outside, inside):

    from ..gui import show_info
    show_info('Found %d unique fits from %d random placements ' %
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
    show_info('%s and times found:\n\t%s\n' % (sname, scores))
    show_info('Best fit found:\n%s' % f0.fit_message())

# -----------------------------------------------------------------------------
#
def fit_sequence(atomsOrMap, volume, metric, envelope, resolution,
                 shift, rotate, moveWholeMolecules, sequence,
                 maxSteps, gridStepMin, gridStepMax):

    if resolution is None:
        from ..VolumeViewer import Volume
        vlist = [v for v in atomsOrMap.models() if isinstance(v, Volume)]
        if len(vlist) < 2:
            if len(atomsOrMap.atoms()) > 0:
                raise CommandError('Fit sequence requires "resolution"'
                                   ' argument to fit atomic models')
            else:
                raise CommandError('Fit sequence requires 2 or more maps'
                                   ' to place')
    else:
        mlist = atomsOrMap.molecules()
        if len(mlist) < 2:
            raise CommandError('Fit sequence requires 2 or more molecules')
        if not moveWholeMolecules:
            raise CommandError('Fit sequence does not support'
                               ' moving partial molecules')
            # TODO: Handle case where not moving whole molecules.
        import fitmap as F
        vlist = [F.simulated_map(m.atoms, resolution, moveWholeMolecules)
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

  from ..VolumeViewer import Volume
  vlist = [m for m in selection.models() if isinstance(m, Volume)]
  if len(vlist) == 0:
    raise CommandError('No atoms or maps for %s' % selection.oslStr)
  elif len(vlist) > 1:
    raise CommandError('Multiple maps for %s' % selection.oslStr)
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
    raise CommandError('metric "%s" must be "overlap" or "correlation"' % metric)
  return me

# -----------------------------------------------------------------------------
# Returns points in global coordinates.
#
def map_fitting_points(v, envelope, local_coords = False):

    from ..place import identity
    point_to_scene_transform = None if local_coords else identity()
    from . import fitmap as F
    try:
        points, point_weights = F.map_points_and_weights(v, envelope,
                                                         point_to_scene_transform)
    except (MemoryError, ValueError) as e:
        raise CommandError('Out of memory, too many points in %s' % v.name)

    if len(points) == 0:
        msg = ('No grid points above map threshold.' if envelope
               else 'Map has no non-zero values.')
        raise CommandError(msg)

    return points, point_weights

# -----------------------------------------------------------------------------
#
def report_status(message):

  from .. import gui
  gui.show_status(message)
  return False        # Don't halt computation
