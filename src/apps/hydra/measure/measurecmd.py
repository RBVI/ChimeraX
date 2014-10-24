# -----------------------------------------------------------------------------
# Command to calculate and display various measurements.
#
#   Syntax: measure <operation> <params>
#
# where operation and parameters are one of
#
#       rotation <mspec1> <mspec2> [showAxis True] [color blue]
#
#       volume <sspec>
#
#       area <sspec>
#
#       correlation <vspec1> <vspec2> [aboveThreshold True|False]
#                       [rotationAxis <spec>] [angleRange 0,360,2] [plot True]
#
#       buriedarea <aspec1> <aspec2> [probeRadius 1.4] [vertexDensity 2.0]
#
def measure_command(cmd_name, args, session):

    from ..commands.parse import specifier_arg, bool_arg, color_256_arg, model_id_int_arg
    from ..commands.parse import surfaces_arg, float_arg, volume_arg, no_arg, perform_operation
    ops = {
        'inertia': (inertia,
                (('objects', specifier_arg),),
                (),
                (('showEllipsoid', bool_arg),
                 ('color', color_256_arg),
                 ('perChain', bool_arg),
                 ('modelId', model_id_int_arg),
                 ('replace', bool_arg),)),
        'motion': (motion_lines,
                   (('surfaces', surfaces_arg),),
                   (),
                   (('length', float_arg),
                    ('color', color_256_arg),
                    ('toward', volume_arg),
                    ('scale', float_arg),
                    ('modelId', model_id_int_arg),
                    ('replace', bool_arg),)),
    }
    perform_operation(cmd_name, args, ops, session)

# Old measure command code from Chimera 1
def old_measure_command(cmdname, args):
    from Commands import CommandError

    operations = {'rotation': (rotation_axis, [('model1Spec','model1','models'),
                                             ('model2Spec','model2','models')]),
                  'volume': (volume_area, [('surfaceSpec','surface','vertices')]),
                  'area': (volume_area, [('surfaceSpec','surface','vertices')]),
                  'correlation': (correlation, [('map1Spec','map1','models'),
                                                ('map2Spec','map2','models')]),
                  'buriedArea': (buried_area, [('a1Spec','atoms1','atoms'),
                                                ('a2Spec','atoms2','atoms')]),
                  'contactArea': (contact_area, []),
                  'inertia': (inertia, []),
                  'spine': (spine, []),
                  'distance': (distance,
                               [('object1Spec','object1',None),
                                ('object2Spec','object2',None)]),
                  'mapValues': (map_values, [('mapSpec', 'volume', 'models'),
                                             ('atomSpec','atoms','atoms')]),
                  'pathLength': (path_length, [('pathSpec', 'path', None)]),
                  'symmetry': (symmetry, [('volumeSpec','volume','models')]),
                  'center': (center, [('objectSpec','objects',None)]),
                  'fieldLines': (field_lines, [('volumeSpec','volume','models')]),
                  'mapStats': (map_statistics, [('volumeSpec','volume','models')]),
                  'mapSum': (map_sum, [('volumeSpec','volume','models')]),
                  }
    from Commands import abbreviation_table
    aop = abbreviation_table(operations.keys())

    fields = args.split(' ', 1)
    if len(fields) == 0:
        ops = ', '.join(operations.keys())
        raise CommandError('Missing required argument: "operation" (%s)' % ops)
    op = aop.get(fields[0].lower())
    if op is None:
        ops = ', '.join(operations)
        raise CommandError('Unknown operation "%s" (use %s)' % (fields[0], ops))
        
    from Commands import doExtensionFunc
    f, specInfo = operations[op]
    doExtensionFunc(f, args, specInfo = specInfo)

# -----------------------------------------------------------------------------
#
def rotation_axis(operation, model1, model2,
                  showAxis = True, showSlabs = False, color = None,
                  coordinateSystem = None):

    os1 = set([m.openState for m in model1])
    os2 = set([m.openState for m in model2])
    if len(os1) != 1:
        raise CommandError('First model spec names %d models, require 1' % len(os1))
    if len(os2) != 1:
        raise CommandError('Second model spec names %d models, require 1' % len(os2))
    os1 = os1.pop()
    os2 = os2.pop()

    xf = os1.xform.inverse()
    xf.multiply(os2.xform)
    import Matrix
    tf = Matrix.xform_matrix(xf)
    if coordinateSystem:
        import Commands
        csys = Commands.openstate_arg(coordinateSystem)
        cxf = csys.xform
        cxf.premultiply(os1.xform.inverse())
        tf = Matrix.coordinate_transform(tf, Matrix.xform_matrix(cxf))
    else:
        csys = os1

    m1, m2, cm = model1[0], model2[0], csys.models[0]
    if csys == os1:
        message = ('Position of %s (%s) relative to %s (%s) coordinates:\n'
                   % (m2.name, m2.oslIdent(), m1.name, m1.oslIdent()))
    else:
        message = ('Position of %s (%s) relative to %s (%s) in %s (%s) coordinates:\n'
                   % (m2.name, m2.oslIdent(), m1.name, m1.oslIdent(),
                      cm.name, cm.oslIdent()))
    message += Matrix.transformation_description(tf)
    from chimera import replyobj
    replyobj.info(message)
    ra = Matrix.axis_center_angle_shift(tf)[2]
    replyobj.status('Rotation angle %.2f degrees' % ra)

    from chimera import MaterialColor
    if isinstance(color, MaterialColor):
        color = color.rgba()
    elif not color is None:
        raise CommandError('Unknown color "%s"' % str(color))

    if showAxis:
        show_axis(tf, color, csys)

    if showSlabs:
        have_box, box = os1.bbox()
        if have_box:
            center = box.center().data()
            show_slabs(Matrix.chimera_xform(tf), color, center, csys)

# -----------------------------------------------------------------------------
#
def show_axis(tf, color, os):

    import Matrix
    axis, axis_point, angle, axis_shift = Matrix.axis_center_angle_shift(tf)
    if angle < 0.1:
        raise CommandError('Rotation angle is near zero (%g degrees)' % angle)

    have_box, box = os.bbox()
    if not have_box:
        # TODO: Chimera does not provide bounding box of full model.
        raise CommandError('First model must be visible to show axis')

    axis_center = Matrix.project_to_axis(box.center().data(), axis, axis_point)
    axis_length = max((box.urb - box.llf).data())
    hl = 0.5*axis_length
    ap1 = map(lambda a,b: a-hl*b, axis_center, axis)
    ap2 = map(lambda a,b: a+hl*b, axis_center, axis)
    from VolumePath import Marker_Set, Link
    from VolumePath.markerset import chimera_color
    m = Marker_Set('rotation axis')
    mm = m.marker_model()
    mm.openState.xform = os.xform
    if color:
        mm.color = chimera_color(color)
    radius = 0.025 * axis_length
    m1 = m.place_marker(ap1, None, radius)
    m2 = m.place_marker(ap2, None, radius)
    Link(m1, m2, None, radius)

# -----------------------------------------------------------------------------
# xf is xform in os coordinates.
#
def show_slabs(xf, color, center, os):

    # Make schematic illustrating rotation
    if color is None:
        color = (.7,.7,.7,1)
    import MatchDomains
    sm = MatchDomains.transform_schematic(xf, center, color, color)
    if sm:
        sm.name = 'slabs'
        from chimera import openModels as om
        om.add([sm])
        sm.openState.xform = os.xform
    return sm

# -----------------------------------------------------------------------------
#
def volume_area(operation, surface):

    from _surface import SurfacePiece
    plist = [p for p in surface if isinstance(p, SurfacePiece)]
    from Surface import filter_surface_pieces
    fplist = filter_surface_pieces(plist)
    if len(fplist) == 0:
        fplist = filter_surface_pieces(plist, include_outline_boxes = True,
                                       include_surface_caps = True)
        if len(fplist) == 0:
            raise CommandError('No surfaces specified')

    import MeasureVolume as m
    op = operation[0]
    m.report_volume_and_area(fplist,
                             report_volume = (op == 'v'),
                             report_area = (op == 'a'))

# -----------------------------------------------------------------------------
#
def correlation(operation, map1, map2, aboveThreshold = True,
                rotationAxis = None, angleRange = (0,360,2),
                plot = True):

    from VolumeViewer import Volume
    map1 = [m for m in map1 if isinstance(m, Volume)]
    map2 = [m for m in map2 if isinstance(m, Volume)]
    if len(map1) == 0 or len(map2) == 0:
        raise CommandError('Must specify 2 maps')

    if rotationAxis:
        # Rotate map1 in steps and report correlations.
        from Commands import parse_axis
        axis, center, csys = parse_axis(rotationAxis, 'measure')
        if center is None:
            raise CommandError('Rotation axis must be atom/bond spec')
        axis = csys.xform.apply(axis).data()
        center = csys.xform.apply(center).data()
        if isinstance(angleRange, str):
            try:
                angleRange = [float(x) for x in angleRange.split(',')]
            except:
                angleRange = []
            if len(angleRange) != 3:
                raise CommandError('Angle range must be 3 comma-separated values, got "%s"' % angleRange)

        for v1 in map1:
            for v2 in map2:
                report_correlations_with_rotation(v1, v2, aboveThreshold,
                                                  axis, center, angleRange,
                                                  plot)
    else:
        for v1 in map1:
            for v2 in map2:
                report_correlation(v1, v2, aboveThreshold)

# -----------------------------------------------------------------------------
#
def report_correlation(v1, v2, aboveThreshold):
            
    from chimera import replyobj
    import FitMap
    olap, cor, corm = FitMap.map_overlap_and_correlation(v1, v2, aboveThreshold)
    replyobj.status('correlation = %.4g, corr about mean = %.4g' % (cor, corm))
    replyobj.info('Correlation between %s and %s = %.4g, about mean = %.4g\n'
                  % (v1.name, v2.name, cor, corm))

# -----------------------------------------------------------------------------
#
def report_correlations_with_rotation(v1, v2, aboveThreshold,
                                      axis, center, angleRange, plot):

    from chimera import Vector, Point, replyobj

    # Convert axis and center to v1 local coordinates so transformation
    # is still valid if user rotates entire scene.
    xf = v1.openState.xform.inverse()
    axis = xf.apply(Vector(*axis)).data()
    center = xf.apply(Point(*center)).data()

    import FitMap, Matrix
    replyobj.info('Correlation between %s and %s\n' % (v1.name, v2.name) +
                  'Rotation\tCorrelation\tCorr About Mean\n')
    a0, a1, astep = angleRange
    angle = a0
    clist = []
    from Matrix import multiply_matrices, xform_matrix, rotation_transform, chimera_xform
    while angle < a1:
        tf = multiply_matrices(xform_matrix(v1.openState.xform),
                               rotation_transform(axis, angle, center),
                               xform_matrix(v1.openState.xform.inverse()))
        xf = chimera_xform(tf)
        olap, cor, corm = FitMap.map_overlap_and_correlation(v1, v2,
                                                             aboveThreshold, xf)
        replyobj.status('angle = %.4g, correlation = %.4g, corr about mean = %.4f' % (angle, cor, corm))
        replyobj.info('%.4g\t%.4g\t%.4g\n' % (angle, cor, corm))
        clist.append((angle,cor))
        angle += astep

    if plot:
        angles = [a for a,c in clist]
        corr = [c for a,c in clist]
        plot_correlation(angles, corr, v1, v2, axis, center)

# -----------------------------------------------------------------------------
#
def plot_correlation(angles, corr, v1, v2, axis, center):

    # TODO: Make Tk plot window to use Chimera root window as master so it
    #       gets iconified when Chimera iconified.  Unfortunately matplotlib
    #       doesn't provide for setting Tk master.  Need to create our own
    #       Toplevel and embed the canvas as in embedding_in_tk.py example.
    #       Also see matplotlib.backends.backend_tkagg.new_figure_manager().
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(angles, corr, linewidth=1.0)
    ax.set_xlim(xmax = 360.0)
    ax.set_xticks(range(0,361,30))
    ax.set_ylim(ymax = 1.0)
    ax.set_xlabel('rotation angle (degrees)')
    ax.set_ylabel('correlation')
    ax.set_title('Correlation of rotated %s vs %s' % (v1.name, v2.name))
    ax.grid(True)
    ax.cur_position = plt.Line2D((0,0), ax.get_ylim(), linewidth = 1.0, color = 'orange')
    ax.add_line(ax.cur_position)
    def graph_event_cb(event, v1=v1, axis=axis, center=center, cur_angle = [0]):
        if not event.button is None:
            angle = event.xdata
            if angle is None:
                angle = 0       # Click outside graph bounds
            import Matrix
            tf = Matrix.rotation_transform(axis, angle-cur_angle[0], center)
            xf = Matrix.chimera_xform(tf)
            v1.openState.localXform(xf)
            cur_angle[0] = angle
            ax.cur_position.set_xdata((angle,angle))
            ax.figure.canvas.draw()

    fig.canvas.mpl_connect('button_press_event', graph_event_cb)
    fig.canvas.mpl_connect('motion_notify_event', graph_event_cb)
    fig.canvas.manager.show()

# -----------------------------------------------------------------------------
#
def buried_area(operation, atoms1, atoms2,
                probeRadius = 1.4, vertexDensity = 2.0):

    xyzr = atom_xyzr(atoms1 + atoms2)
    n1 = len(atoms1)
    xyzr1 = xyzr[:n1]
    xyzr2 = xyzr[n1:]
    xyzr12 = xyzr

    failed = False
    import MoleculeSurface as ms
    try:
        s1 = ms.xyzr_surface_geometry(xyzr1, probeRadius, vertexDensity)
        s2 = ms.xyzr_surface_geometry(xyzr2, probeRadius, vertexDensity)
        s12 = ms.xyzr_surface_geometry(xyzr12, probeRadius, vertexDensity)
    except ms.Surface_Calculation_Error:
        failed = True

    if failed or not s1[6] or not s2[6] or not s12[6]:
        # All component calculation failed.
        try:
            s1 = ms.xyzr_surface_geometry(xyzr1, probeRadius, vertexDensity,
                                          all_components = False)
            s2 = ms.xyzr_surface_geometry(xyzr2, probeRadius, vertexDensity,
                                          all_components = False)
            s12 = ms.xyzr_surface_geometry(xyzr12, probeRadius, vertexDensity,
                                           all_components = False)
        except ms.Surface_Calculation_Error:
            raise CommandError('Surface calculation failed.')
        from chimera import replyobj
        replyobj.warning('Calculation of some surface components failed.  Using only single surface component.  This may give inaccurate areas if surfaces of either set of atoms or the combined set are disconnected.\n')

    # Assign per-atom buried areas.
    aareas1, aareas2, aareas12 = s1[3], s2[3], s12[3]
    ases121 = asas121 = ases122 = asas122 = 0
    for ai,a in enumerate(atoms1):
        a.buriedSESArea = aareas1[ai,0] - aareas12[ai,0]
        a.buriedSASArea = aareas1[ai,1] - aareas12[ai,1]
        ases121 += aareas12[ai,0]
        asas121 += aareas12[ai,1]
    for ai,a in enumerate(atoms2):
        a.buriedSESArea = aareas2[ai,0] - aareas12[n1+ai,0]
        a.buriedSASArea = aareas2[ai,1] - aareas12[n1+ai,1]
        ases122 += aareas12[n1+ai,0]
        asas122 += aareas12[n1+ai,1]
    careas1, careas2, careas12 = s1[4], s2[4], s12[4]
    ases1, asas1 = area_sums(careas1)
    ases2, asas2 = area_sums(careas2)
    ases12, asas12 = area_sums(careas12)
    bsas1 = asas1 - asas121
    bsas2 = asas2 - asas122
    bsas = 0.5 * (bsas1 + bsas2)
    bses1 = ases1 - ases121
    bses2 = ases2 - ases122
    bses = 0.5 * (bses1 + bses2)

    # TODO: include atomspec's in output message.
    msg = ('Buried solvent accessible surface area\n'
           '  B1SAS = %.6g, B2SAS = %.6g, BaveSAS = %.6g\n'
           '  (A1 = %.6g, A2 = %.6g, A12 = %.6g = %.6g + %.6g)\n' %
           (bsas1, bsas2, bsas,asas1, asas2, asas12, asas121, asas122) + 
           'Buried solvent excluded surface area\n ' +
           '  B1SES = %.6g, B2SES = %.6g, BaveSES = %.6g\n'
           '  (A1 = %.6g, A2 = %.6g, A12 = %.6g = %.6g + %.6g)\n' %
           (bses1, bses2, bses,ases1, ases2, ases12, ases121, ases122))
    from chimera import replyobj
    replyobj.info(msg)

    smsg = 'Buried areas: SAS = %.6g, SES = %.6g\n' % (bsas, bses)
    replyobj.status(smsg)

# -----------------------------------------------------------------------------
#
def atom_xyzr(atoms):

    n = len(atoms)
    from numpy import zeros, float32
    xyzr = zeros((n,4), float32)
    for a in range(n):
        atom = atoms[a]
        xyz = atom.xformCoord().data()
        r = atom.radius
        xyzr[a,:] = xyz + (r,)
    return xyzr

# -----------------------------------------------------------------------------
#
def area_sums(careas):

    if len(careas) == 0:
        return 0,0
    import numpy
    return numpy.sum(careas, axis = 0)


# -----------------------------------------------------------------------------
#
def contact_area(operation, surf1, surf2, distance, show = True,
                 color = (1,0,0,1), offset = 1,
                 slab = None, smooth = False, optimize = True):

    plist = []
    import Surface
    for spec in (surf1, surf2):
        s = parse_object_specifier(spec, 'surface')
        p = Surface.selected_surface_pieces(s, include_outline_boxes = False)
        if len(p) == 0:
            raise CommandError('%s has no surface pieces' % spec)
        elif len(p) > 1:
            raise CommandError('%s has %d surface pieces, require 1'
                               % (spec, len(p)))
        plist.append(p[0])
    p1, p2 = plist

    from Commands import parse_color
    color = parse_color(color)
    if not show:
        color = None

    from Commands import check_number
    check_number(offset, 'offset')

    if not slab is None:
        if isinstance(slab, (float,int)):
            slab = (-0.5*slab, 0.5*slab)
        else:
            from Commands import parse_floats
            slab = parse_floats(slab, 'slab', 2)
        offset = None
        
    import contactarea as c
    area = c.contact_area(p1, p2, distance, color, offset, slab, smooth,
                          optimize)

    from chimera import replyobj
    replyobj.info('Contact area on %s within distance %.4g\nof %s = %.4g\n'
                  % (p1.model.name, distance, p2.model.name, area))
    replyobj.status('Contact area = %.4g' % area)
    
# -----------------------------------------------------------------------------
#
def parse_object_specifier(spec, name = 'object'):

    from chimera import specifier
    try:
        sel = specifier.evalSpec(spec)
    except:
        raise CommandError('Bad %s specifier "%s"' % (name, spec))
    return sel

# -----------------------------------------------------------------------------
#
def inertia(objects, showEllipsoid = True, color = None, perChain = False,
            modelId = None, replace = True, session = None):

    from . import inertia

    atoms = objects.atom_set()
    if atoms:
        mols = atoms.molecules()
        mname = molecules_name(mols)
        sname = ('ellipsoids ' if perChain else 'ellipsoid ') + mname
        surf = surface_model(sname, mols[0].position, modelId, replace, session) if showEllipsoid else None
        if perChain:
            catoms = atoms.separate_chains()
            for cat in catoms:
                cid = cat.chains()[0][1] # Chain id
                info = inertia.atoms_inertia_ellipsoid(cat, color, surf)
                session.show_info('Inertia axes for %s, chain %s, %d atoms\n%s'
                                  % (mname, cid, len(cat), info))
        else:
            info = inertia.atoms_inertia_ellipsoid(atoms, color, surf)
            session.show_info('Inertia axes for %s, %d atoms\n%s' % (mname, len(atoms), info))

    surfs = objects.surfaces()
    if surfs:
        sname = surfs[0].name if len(surfs) == 1 else ('%d surfaces' % len(surfs))
        surf = surface_model(sname, surfs[0].position, modelId, replace, session) if showEllipsoid else None
        info = inertia.surface_inertia_ellipsoid(surfs, color, surf)
        session.show_info('Inertia axes for %s\n%s' % (sname, info))

    maps = objects.maps()
    if maps:
        mname = 'ellipsoid ' + (maps[0].name if len(maps) == 1 else ('%d maps' % len(maps)))
        surf = surface_model(mname, maps[0].position, modelId, replace, session) if showEllipsoid else None
        info = inertia.density_map_inertia_ellipsoid(maps, color, surf)
        session.show_info('Inertia axes for %s\n%s' % (mname, info))

# -----------------------------------------------------------------------------
#
def surface_model(name, place, model_id, replace, session):

    if not model_id is None:
        s = session.find_model_by_id(model_id)
        if s:
            if replace:
                session.close_models([s])
            else:
                return s

    from ..models import Model
    s = Model(name)
    s.id = model_id
    s.position = place
    session.add_model(s)
    return s

# -----------------------------------------------------------------------------
#
def motion_lines(surfaces, length = 1, color = (255,255,255,255), toward = None,
                 scale = 1, modelId = None, replace = True, session = None):
    from . import cactus
    for s in surfaces:
        surf = surface_model('motion ' + s.name, s.position, modelId, replace, session)
        if toward is None:
            cactus.show_prickles(s, scale*length, color, surf)     # Just show surface normals
        else:
            cactus.show_motion_lines(s, toward, scale, color, surf)

# -----------------------------------------------------------------------------
#
def spine(operation, regions, spacing = None, tipLength = None, color = None,
          showDiameter = False):

    sel = parse_object_specifier(regions, 'segmentation region')

    import Surface
    plist = Surface.selected_surface_pieces(sel, include_outline_boxes = False)
    from Segger.regions import Segmentation
    rlist = [p.region for p in plist if (hasattr(p,'region') and
                                         isinstance(p.model, Segmentation))]
    if len(rlist) == 0:
        raise CommandError('No segmentation regions specified: "%s"' % regions)

    if not (spacing is None or isinstance(spacing, (int,float))
            and spacing > 0):
        raise CommandError('spacing must be positive numeric value')
    if not (tipLength is None or isinstance(tipLength, (int,float))
            and tipLength > 0):
        raise CommandError('tipLength must be positive numeric value')

    if not color is None:
        from Commands import parse_color
        color = parse_color(color)

    if showDiameter:
        from _surface import SurfaceModel
        diam_model = SurfaceModel()
        diam_model.name = 'Diameters'
        from chimera import openModels
        openModels.add([diam_model], sameAs = rlist[0].segmentation)
    else:
        diam_model = None
        
    import spine
    from chimera import replyobj
    from PathLength import path_length
    for r in rlist:
        mset = spine.trace_spine(r, spacing, tipLength, color)
        slen = path_length([l.bond for l in mset.links()])
        r.set_attribute('spine length', slen)
        msg = 'Spine length for region %d is %.4g' % (r.rid, slen)
        dmax, dmin = spine.measure_diameter(r, mset, diam_model)
        if not dmax is None:
            r.set_attribute('diameter1', dmax)
            r.set_attribute('diameter2', dmin)
            msg += ', diameters %.4g, %.4g' % (dmax, dmin)
        kave, kmin, kmax = spine.measure_curvature(mset)
        if not kmax is None:
            r.set_attribute('curvature average', kave)
            r.set_attribute('curvature minimum', kmin)
            r.set_attribute('curvature maximum', kmax)
            msg += ', curvature %.4g (ave), %.4g (max), %.4g (min)' % (kave, kmax, kmin)
        replyobj.info(msg + '\n')

# ----------------------------------------------------------------------------
#        
def molecules_name(mlist):

    if len(mlist) == 1:
        return mlist[0].name
    return '%d molecules' % len(mlist)

# -----------------------------------------------------------------------------
#
def distance(operation, object1, object2, multiple = False,
             show = False, color = (0,1,1,1)):

    a1 = object1.atoms()
    import Surface as s
    s1 = s.selected_surface_pieces(object1, include_outline_boxes = False)

    if len(a1) == 0 and len(s1) == 0:
      raise CommandError('No atoms or surfaces specified')

    a2 = object2.atoms()
    s2 = s.selected_surface_pieces(object2, include_outline_boxes = False)

    if len(a2) == 0 and len(s2) == 0:
      raise CommandError('No target atoms or surfaces')

    # Remove near stuff.
    if a1:
      a2 = list(set(a2).difference(a1))
    if s1:
      s2 = list(set(s2).difference(s1))

    name2 = object_name(a2,s2)
    xyz2 = point_array(a2,s2)

    if show:
        from Commands import parse_color
        color = parse_color(color)
        from _surface import SurfaceModel
        surf = SurfaceModel()
        surf.name = 'Distance measurement'
        from chimera import openModels
        openModels.add([surf])
    else:
        surf = None

    if multiple:
        pairs = [([a],[]) for a in a1] + [([],[s]) for s in s1]
    else:
        pairs = [(a1,s1)]

    for a,s in pairs:
        name = object_name(a,s)
        xyz = point_array(a,s)
        report_distance(xyz, xyz2, name, name2, surf, color)

# -----------------------------------------------------------------------------
#
def report_distance(xyz1, xyz2, from_name, to_name, surf, color):

    i1, i2, d = closest_approach(xyz1, xyz2)

    msg = 'minimum distance from %s to %s = %.5g' % (from_name, to_name, d)
    from chimera import replyobj
    replyobj.status(msg)
    replyobj.info(msg + '\n')

    if surf:
        from Matrix import apply_matrix, xform_matrix
        tf = xform_matrix(surf.openState.xform.inverse())
        v = apply_matrix(tf, (xyz1[i1],xyz2[i2]))
        t = [(0,1,0)]
        p = surf.addPiece(v, t, color)
        p.displayStyle = p.Mesh
        p.useLighting = False
        p.lineThickness = 3

# -----------------------------------------------------------------------------
#
def object_name(atoms, surfs):

    na,ns = len(atoms), len(surfs)
    if na > 0 and ns > 0:
        name = '%d atoms, %d surfaces' % (na,ns)
    elif na > 1:
        name = '%d atoms' % na
    elif ns > 1:
        name = '%d surfaces' % ns
    elif na > 0:
        name = atoms[0].oslIdent()
    elif ns > 0:
        name = surfs[0].oslIdent()
    else:
        name = 'nothing'
        
    return name

# -----------------------------------------------------------------------------
# Place atom coordinates and surface vertices in a single array in global
# coordinates.
#
def point_array(atoms, surfs):

    na = len(atoms)
    n = na + sum([s.vertexCount for s in surfs])
    from numpy import empty, float32
    xyz = empty((n,3), float32)
    for i,a in enumerate(atoms):
        xyz[i] = a.xformCoord().data()
    from Matrix import xform_points
    o = na
    for s in surfs:
        xyz[o:o+s.vertexCount] = s.geometry[0]
        xform_points(xyz[o:o+s.vertexCount], s.model.openState.xform)
        o += s.vertexCount
    return xyz

# -----------------------------------------------------------------------------
#
def closest_approach(xyz1, xyz2):

    dmax = minimum_distance_upper_bound(xyz1, xyz2)
    from _closepoints import find_closest_points, BOXES_METHOD
    i1, i2, i2close = find_closest_points(BOXES_METHOD, xyz1, xyz2, dmax)
    v = xyz1[i1,:] - xyz2[i2close,:]
    d2 = (v*v).sum(axis = 1)
    i = d2.argmin()
    from math import sqrt
    return i1[i], i2close[i], sqrt(d2[i])

# -----------------------------------------------------------------------------
# Important to get a small upper bound for efficient calculation of
# closest approach for large sets.
#
def minimum_distance_upper_bound(xyz1, xyz2):

    n = 1000
    if len(xyz1) > n and len(xyz2) > n:
        from numpy.random import randint
        p1 = xyz1[randint(0,len(xyz1),size=(n,)),:]
        p2 = xyz2[randint(0,len(xyz2),size=(n,)),:]
        dmax = closest_approach(p1, p2)[2]
    else:
        from numpy.linalg import norm
        dmax = norm(xyz1[0] - xyz2[0])
    return 1.001*dmax

# -----------------------------------------------------------------------------
#
def map_values(operation, volume, atoms, name = None, report = 10):

    
    from VolumeViewer import Volume
    vlist = [v for v in volume if isinstance(v, Volume)]
    if len(vlist) != 1:
        raise CommandError('No volume model specified')
    v = vlist[0]

    import AtomDensity
    if name is None:
        name = 'value_' + v.name
        name = AtomDensity.replace_special_characters(name,'_')
    AtomDensity.set_atom_volume_values(atoms, v, name)

    if report:
        arep = atoms[:report] if isinstance(report, int) else atoms
        values = '\n'.join(['%s %.5g' % (a.oslIdent(),getattr(a,name))
                            for a in arep])
        n = len(atoms)
        t = '%s map values at %d atom positions\n%s\n' % (v.name, n, values)
        if n > len(arep):
            t += '...\n'
        if n == 1:
            s = '%s map value at atom %s = %.5g' % (v.name, atoms[0].oslIdent(), getattr(a,name))
        else:
            maxa = 3
            values = ', '.join(['%.5g' % getattr(a,name) for a in atoms[:maxa]])
            if n > maxa:
                values += ', ...'
            s = '%s map values at %d atoms: %s' % (v.name, n, values)
        from chimera import replyobj
        replyobj.info(t)
        replyobj.status(s)

# -----------------------------------------------------------------------------
#
def path_length(operation, path, group = 'all'):

    from chimera import selection
    if isinstance(path, selection.OSLSelection):
        # OSLSelection never includes bonds.  Use bonds between selected atoms.
        atoms = path.contents()[0]
        path = selection.ItemizedSelection(atoms)
        path.addImplied()
        
    bonds = path.bonds()
    if len(bonds) == 0:
        raise CommandError('No bonds specified')

    if group == 'all':
        groups = [bonds]
    elif group == 'models':
        groups = bonds_per_molecule(bonds)
    elif group == 'connected':
        groups = connected_bonds(bonds)
    else:
        raise CommandError('Group must be "all", "models", or "connected"')

    lines = []
    totlen = 0
    import PathLength
    for g in groups:
        length = PathLength.path_length(g)
        totlen += length
        lines.append('%s ... %s path, %d segments, length = %.5g' %
                     (g[0].atoms[0].oslIdent(), g[-1].atoms[1].oslIdent(),
                      len(g), length))
    avelen = totlen / len(groups)

    t = '\n'.join(lines) + '\n'
    s = lines[0] if len(lines) == 1 else '%d paths, average length = %.5g' % (len(groups), avelen)
    from chimera import replyobj
    replyobj.info(t)
    replyobj.status(s)

# -----------------------------------------------------------------------------
#
def bonds_per_molecule(bonds):

    mb = {}
    for b in bonds:
        m = b.molecule
        if m in mb:
            mb[m].append(b)
        else:
            mb[m] = [b]
    return mb.values()

# -----------------------------------------------------------------------------
# Return list of groups of connected bonds, each group being a list of bonds.
#
def connected_bonds(bonds):

    ca = {}     # Map atoms to list of connected atoms
    for b in bonds:
        a1,a2 = b.atoms
        f1 = a1 in ca
        f2 = a2 in ca
        if f1 and not f2:
            ca[a1].append(a2)
            ca[a2] = ca[a1]
        elif not f1 and f2:
            ca[a2].append(a1)
            ca[a1] = ca[a2]
        elif f1 and f2:
            if not ca[a1] is ca[a2]:
                alist = ca[a1] + ca[a2]
                for a in alist:
                    ca[a] = alist
        else:
            ca[a1] = ca[a2] = [a1,a2]
    cb = {}
    for b in bonds:
        i = id(ca[b.atoms[0]])
        if i in cb:
            cb[i].append(b)
        else:
            cb[i] = [b]
    groups = cb.values()
    return groups

# -----------------------------------------------------------------------------
#
def symmetry(operation, volume, minimumCorrelation = 0.99, nMax = 8,
             helix = None, points = 10000, set = True):

    from VolumeViewer import Volume
    vlist = [m for m in volume if isinstance(m, Volume)]
    if len(vlist) == 0:
        raise CommandError('No volume specified')

    if not helix is None:
        rise, angle, n, optimize = parse_helix_option(helix)

    import symmetry as S
    for v in vlist:
        if helix:
            syms, msg = S.find_helix_symmetry(v, rise, angle, n, optimize, nMax,
                                              minimumCorrelation, points)
        else:
            syms, msg = S.find_point_symmetry(v, nMax,
                                              minimumCorrelation, points)

        if set and syms:
            v.data.symmetries = syms

        from chimera.replyobj import info, status
        status(msg)
        info(msg + '\n')

# -----------------------------------------------------------------------------
#
def parse_helix_option(helix):

    herr = 'Invalid helix option <rise>,<angle>[,<n>][,opt]'
    if not isinstance(helix, str):
        raise CommandError(herr)
    fields = helix.split(',')
    optimize = (fields and fields[-1] == 'opt')
    if optimize:
        fields = fields[:-1]
    if len(fields) in (2,3):
        try:
            rise, angle = [float(f) for f in fields[:2]]
            n = int(fields[2]) if len(fields) == 3 else None
        except ValueError:
            raise CommandError(herr)
    else:
        raise CommandError(herr)
    return rise, angle, n, optimize

# -----------------------------------------------------------------------------
#
def center(operation, objects, level = None, mark = False, color = (.7,.7,.7,1),
           radius = None, name = None, modelId = None):

    mlist = objects.models()
    from VolumeViewer import Volume
    vlist = [m for m in mlist if isinstance(m, Volume)]
    atoms = objects.atoms()
    if len(vlist) == 0 and len(atoms) == 0:
        raise CommandError('No volume or atoms specified')

    import Commands as CD
    rgba = CD.parse_color(color)
    model_id = None if modelId is None else CD.parse_model_id(modelId)

    import center as C
    from chimera import replyobj
    for v in vlist:
        ijk = C.volume_center_of_mass(v, level)
        xyz = v.data.ijk_to_xyz(ijk)
        msg = ('Center of mass grid index for %s = (%.2f, %.2f, %.2f)'
               % (v.name, ijk[0], ijk[1], ijk[2]))
        replyobj.info(msg + '\n')
        replyobj.status(msg)
        if mark:
            r = max(v.data.step) if radius is None else radius
            mname = v.name + ' center' if name is None else name
            C.place_marker(xyz, v, rgba, r, mname, model_id)

    if len(atoms) > 0:
        m0 = atoms[0].molecule
        cxf = m0.openState.xform
        xyz = C.atoms_center_of_mass(atoms, cxf)
        msg = ('Center of mass of %d atoms in %s (%s) coordinate system'
               ' = (%.2f, %.2f, %.2f)'
               % (len(atoms), m0.name, m0.oslIdent(), xyz[0], xyz[1], xyz[2]))
        replyobj.info(msg + '\n')
        smsg = ('Center for %d atoms = (%.2f, %.2f, %.2f)'
                % (len(atoms), xyz[0], xyz[1], xyz[2]))
        replyobj.status(smsg)
        if mark:
            r = atoms[0].radius if radius is None else radius
            mname = C.atoms_center_model_name(atoms) if name is None else name
            C.place_marker(xyz, m0, rgba, r, mname, model_id)

# -----------------------------------------------------------------------------
#
def field_lines(operation, volume, lines = 1000, startAbove = None, step = 0.5,
                color = (.7,.7,.7,1), lineWidth = 1, tubeRadius = None,
                circleSubdivisions = 12, markers = False, modelId = None):

    from VolumeViewer import Volume
    vlist = [m for m in volume if isinstance(m, Volume)]
    if len(vlist) == 0:
        raise CommandError('No volume specified')

    import Commands as CD
    rgba = CD.parse_color(color)
    model_id = None if modelId is None else CD.parse_model_id(modelId)

    import fieldlines
    for v in vlist:
        fieldlines.show_field_lines(v, lines, startAbove, step,
                                    rgba, lineWidth, tubeRadius,
                                    circleSubdivisions, markers, model_id)

# -----------------------------------------------------------------------------
#
def map_statistics(operation, volume, step = 1, subregion = 'all'):

    from VolumeViewer import Volume
    vlist = [m for m in volume if isinstance(m, Volume)]
    if len(vlist) == 0:
        raise CommandError('No volume specified')

    from Commands import parse_step, parse_subregion
    subregion_arg = subregion
    subregion = parse_subregion(subregion)
    step_arg = step
    step = parse_step(step)

    import VolumeStatistics as VS
    for v in vlist:
        m = v.matrix(step = step, subregion = subregion)
        mean, sd, rms = VS.mean_sd_rms(m)
        descrip = v.name
        if step_arg != 1:
            descrip += ', step %s' % step_arg
        if subregion_arg != 'all':
            descrip += ', subregion %s' % subregion_arg
        msg = '%s: mean = %.5g, SD = %.5g, RMS = %.5g' % (descrip, mean, sd, rms)
        VS.message(msg, show_reply_log = True)

# -----------------------------------------------------------------------------
#
def map_sum(operation, volume, aboveThreshold = None, step = 1, subregion = 'all'):

    from VolumeViewer import Volume
    vlist = [m for m in volume if isinstance(m, Volume)]
    if len(vlist) == 0:
        raise CommandError('No volume specified')

    from Commands import parse_step, parse_subregion
    subregion_arg = subregion
    subregion = parse_subregion(subregion)
    step_arg = step
    step = parse_step(step)

    import numpy
    import VolumeStatistics as VS
    for v in vlist:
        m = v.matrix(step = step, subregion = subregion)
        if aboveThreshold is None:
            s = m.sum(dtype = numpy.float64)
        else:
            ma = (m >= aboveThreshold)
            na = ma.sum(dtype = numpy.int64)
            mt = ma.astype(m.dtype)
            mt *= m
            s = mt.sum(dtype = numpy.float64)
        descrip = v.name
        if step_arg != 1:
            descrip += ', step %s' % step_arg
        if subregion_arg != 'all':
            descrip += ', subregion %s' % subregion_arg
        if not aboveThreshold is None:
            descrip += ', level >= %.5g, npoints %d' % (aboveThreshold, na)
        msg = '%s: sum = %.5g' % (descrip, s)
        VS.message(msg, show_reply_log = True)
