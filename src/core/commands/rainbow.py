def rainbow(session, objects, sequential='residues', target=None, transparency=None,
            cmap=None, cmap_range=None, halfbond=None):
    '''
    Color residues or chains by sequence using a color map.
    Arguments are the same as for the color command.
    '''
    from .color import color
    color(session, objects, target=target, transparency=transparency,
          sequential=sequential, cmap=cmap, cmap_range=cmap_range, halfbond=halfbond)

# -----------------------------------------------------------------------------
#
def register_command(session):
    from . import register, CmdDesc, ColormapArg, ObjectsArg
    from . import EmptyArg, Or, EnumOf, StringArg, TupleOf, FloatArg, BoolArg
    from .color import _SequentialLevels, _CmapRanges
    desc = CmdDesc(required=[('objects', Or(ObjectsArg, EmptyArg))],
                   optional=[('sequential', EnumOf(_SequentialLevels))],
                   keyword=[('target', StringArg),
                            ('transparency', FloatArg),
                            ('cmap', ColormapArg),
                            ('cmap_range', Or(TupleOf(FloatArg, 2), EnumOf(_CmapRanges))),
                            ('halfbond', BoolArg)],
                   synopsis="color residues and chains sequentially")
    register('rainbow', desc, rainbow)
