# -----------------------------------------------------------------------------
#
def register_color_subcommand(command_name, logger):
    from chimerax.core.commands import register, CmdDesc, ColormapArg, ColormapRangeArg
    from chimerax.core.commands import FloatArg, BoolArg
    from chimerax.core.commands import SurfacesArg, CenterArg, AxisArg, CoordSysArg
    from chimerax.map import MapArg

    from .colorvol import color_electrostatic, color_sample, color_gradient
    from .colorgeom import color_radial, color_cylindrical, color_height
    
    map_args = [('map', MapArg),
                ('palette', ColormapArg),
                ('range', ColormapRangeArg),
                ('key', BoolArg),
                ('offset', FloatArg),
                ('transparency', FloatArg),
                ('update', BoolArg),
    ]
    
    geom_args = [('center', CenterArg),
                 ('coordinate_system', CoordSysArg),
                 ('palette', ColormapArg),
                 ('range', ColormapRangeArg),
                 ('transparency', FloatArg),
                 ('key', BoolArg),
                 ('update', BoolArg),
    ]

    if command_name == 'color electrostatic':
        # color by electrostatic potential map 
        desc = CmdDesc(required=[('surfaces', SurfacesArg)],
                       keyword=map_args,
                       required_arguments = ['map'],
                       synopsis="color surfaces by electrostatic potential map value")
        register('color electrostatic', desc, color_electrostatic, logger=logger)

    elif command_name == 'color sample':
        # color by map value
        desc = CmdDesc(required=[('surfaces', SurfacesArg)],
                       keyword=map_args,
                       required_arguments = ['map'],
                       synopsis="color surfaces by map value")
        register('color sample', desc, color_sample, logger=logger)

    elif command_name == 'color gradient':
        # color by map gradient norm
        desc = CmdDesc(required=[('surfaces', SurfacesArg)],
                       keyword=map_args,
                       synopsis="color surfaces by map gradient norm")
        register('color gradient', desc, color_gradient, logger=logger)

    elif command_name == 'color radial':
        # color by radius
        desc = CmdDesc(required=[('surfaces', SurfacesArg)],
                       keyword=geom_args,
                       synopsis="color surfaces by radius")
        register('color radial', desc, color_radial, logger=logger)

    elif command_name == 'color cylindrical':
        # color by cylinder radius
        desc = CmdDesc(required=[('surfaces', SurfacesArg)],
                       keyword=geom_args + [('axis', AxisArg)],
                       synopsis="color surfaces by cylinder radius")
        register('color cylindrical', desc, color_cylindrical, logger=logger)
    elif command_name == 'color height':
        # color by height
        desc = CmdDesc(required=[('surfaces', SurfacesArg)],
                       keyword=geom_args + [('axis', AxisArg)],
                       synopsis="color surfaces by distance along an axis")
        register('color height', desc, color_height, logger=logger)
    elif command_name == 'color image':
        from . import texture
        texture.register_color_image_command(logger)
