# Script to align each volume with the previous one saving to a new map file.
# Also shift mean value to zero.
# Also scale so that specified enclosed volume is at a specified threshold level.

def align_series(maps, path, align = True, zero_mean = True,
                 enclosed_volume = None, enclosed_volume_threshold = None,
                 compress = False):

    from numpy import float32
    v0 = maps[0].writable_copy(value_type = float32)
    m = v0.full_matrix()
    vprev = None
    for i,v in enumerate(maps):
        from chimera.replyobj import status
        status('Processing %s (%d of %d)' % (v.name, i+1, len(maps)))
        if align and vprev:
            v.openState.xform = vprev.openState.xform
            from FitMap.fitmap import map_points_and_weights, motion_to_maximum
            points, point_weights = map_points_and_weights(v, above_threshold = True)
            move_tf, stats = motion_to_maximum(points, point_weights, vprev,
                                               max_steps = 2000,
                                               ijk_step_size_min = 0.01,
                                               ijk_step_size_max = 0.5,
                                               optimize_translation = True,
                                               optimize_rotation = True)
            import Matrix
            v.openState.globalXform(Matrix.chimera_xform(move_tf))
        vprev = v
        m[:,:,:] = 0
        v0.add_interpolated_values(v)

        if zero_mean:
            from numpy import float64
            m -= m.mean(dtype = float64)

        if not enclosed_volume is None and not enclosed_volume_threshold is None:
            level = v0.surface_level_for_enclosed_volume(enclosed_volume)
            m *= float(enclosed_volume_threshold) / level

        v0.data.values_changed()
        v0.data.name = v.data.name
        options = {'append': True, 'compress': compress}
        from VolumeData import cmap
        cmap.write_grid_as_chimera_map(v0.data, path, options)

def alignseries_command(cmdname, args):

    from Commands import bool_arg, float_arg, string_arg, volumes_arg, parse_arguments
    req_args = (('maps', volumes_arg),
                ('path', string_arg))
    opt_args = ()
    kw_args = (('align', bool_arg),
               ('zero_mean', bool_arg),
               ('enclosed_volume', float_arg),
               ('enclosed_volume_threshold', float_arg),
               ('compress', bool_arg))
    kw = parse_arguments(cmdname, args, req_args, opt_args, kw_args)
    align_series(**kw)

from Midas import midas_text as mt
mt.addCommand('alignseries', alignseries_command, None)

# To align a second channel to match alignment of first
# perframe "matrixcopy #$1 #$2" frames 150 range 0,149 range 150,299

# Cell 6 processing commands
#
# vol #0-299 voxelsize .1,.1,.25
# alignseries #0-149 /Users/goddard/Desktop/cell6_align_ch0.cmap enclosed_volume 1000 enclosed_volume_threshold 100
# perframe "matrixcopy #$1 #$2" frames 150 range 0,149 range 150,299
# alignseries #150-299 /Users/goddard/Desktop/cell6_align_ch1.cmap align false enclosed_volume 500 enclosed_volume_threshold 100
#
#
# vop morph #0-149 playstep 0.000809 interp false frames 1236 model #300 ; vop morph #150-299 playstep 0.000809 interp false frames 1236 model #301 ; movie record supersample 3 ; wait 1250 ; movie encode ~/Desktop/cell6_5x.mp4 quality higher framerate 30
