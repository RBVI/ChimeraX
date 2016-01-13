# vim: set expandtab shiftwidth=4 softtabstop=4:


def register_core_commands(session):
    """Register core commands"""
    from importlib import import_module
    # Remember that the order is important, when a command name is
    # abbreviated, the first one registered that matches wins, not
    # the first in alphabetical order.
    modules = [
        'alias', 'buriedarea',
        'camera', 'clip', 'close', 'cofr', 'color', 'colorname', 'crossfade', 'crosslinks',
        'delete', 'echo', 'exit', 'export', 'hide', 'info',
        'lighting', 'list', 'material', 'mousemode', 'move',
        'open', 'pdbimages', 'perframe', 'position', 'pwd', 'rainbow', 'roll', 'run',
        'save', 'sasa', 'scolor', 'select', 'set', 'show', 'split',
        'stop', 'style', 'surface', 'sym',
        'time', 'transparency', 'turn',
        'usage', 'view', 'version', 'wait', 'windowsize', 'zoom'
    ]
    for mod in modules:
        m = import_module(".%s" % mod, __package__)
        m.register_command(session)

    from .. import map
    map.register_volume_command()
    map.register_molmap_command()
    from ..map import filter
    filter.register_vop_command()
    from ..map import fit
    fit.register_fitmap_command()
    from ..map import series
    series.register_vseries_command()

    from ..devices import oculus
    oculus.register_oculus_command()
    from ..devices import spacenavigator
    spacenavigator.register_snav_command()
