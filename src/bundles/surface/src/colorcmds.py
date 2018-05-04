# -----------------------------------------------------------------------------
#
def register_color_subcommands(logger):
    from chimerax.core.commands import register, CmdDesc, ColormapArg, ColormapRangeArg
    from chimerax.core.commands import FloatArg, BoolArg, AtomsArg
    from chimerax.core.commands import SurfacesArg, CenterArg, AxisArg, CoordSysArg
    from chimerax.map import MapArg

    from .colorvol import color_electrostatic, color_sample, color_gradient
    from .colorgeom import color_radial, color_cylindrical, color_height
    from .colorzone import color_zone
    
    map_args = [('map', MapArg),
                ('palette', ColormapArg),
                ('range', ColormapRangeArg),
                ('offset', FloatArg),
                ('transparency', FloatArg),
                ('update', BoolArg),
    ]
    # color by electrostatic potential map 
    desc = CmdDesc(required=[('surfaces', SurfacesArg)],
                   keyword=map_args,
                   required_arguments = ['map'],
                   synopsis="color surfaces by electrostatic potential map value")
    register('color electrostatic', desc, color_electrostatic, logger=logger)

    # color by map value
    desc = CmdDesc(required=[('surfaces', SurfacesArg)],
                   keyword=map_args,
                   required_arguments = ['map'],
                   synopsis="color surfaces by map value")
    register('color sample', desc, color_sample, logger=logger)

    # color by map gradient norm
    desc = CmdDesc(required=[('surfaces', SurfacesArg)],
                   keyword=map_args,
                   required_arguments = ['map'],
                   synopsis="color surfaces by map gradient norm")
    register('color gradient', desc, color_gradient, logger=logger)
    
    # color by radius
    geom_args = [('center', CenterArg),
                 ('coordinate_system', CoordSysArg),
                 ('palette', ColormapArg),
                 ('range', ColormapRangeArg),
                 ('update', BoolArg),
    ]
    desc = CmdDesc(required=[('surfaces', SurfacesArg)],
                   keyword=geom_args,
                   synopsis="color surfaces by radius")
    register('color radial', desc, color_radial, logger=logger)

    # color by cylinder radius
    desc = CmdDesc(required=[('surfaces', SurfacesArg)],
                   keyword=geom_args + [('axis', AxisArg)],
                   synopsis="color surfaces by cylinder radius")
    register('color cylindrical', desc, color_cylindrical, logger=logger)

    # color by height
    desc = CmdDesc(required=[('surfaces', SurfacesArg)],
                   keyword=geom_args + [('axis', AxisArg)],
                   synopsis="color surfaces by distance along an axis")
    register('color height', desc, color_height, logger=logger)
