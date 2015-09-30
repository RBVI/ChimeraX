# vi: set expandtab shiftwidth=4 softtabstop=4:


def initialize(command_name):
    from chimera.core.commands import register
    from chimera.core.commands import CmdDesc, AtomSpecArg, Or, Bounded, FloatArg, EnumOf
    if command_name.startswith('~'):
        from .cartoon import uncartoon
        desc = CmdDesc(optional=[("spec", AtomSpecArg)],
                       synopsis='undisplay cartoon for specified residues')
        register(command_name, desc, uncartoon)
    else:
        from .cartoon import cartoon
        desc = CmdDesc(optional=[("spec", AtomSpecArg),
                                 ("adjust", Or(Bounded(FloatArg, 0.0, 1.0),
                                               EnumOf(["default"]))),
                                 ("style", EnumOf(["ribbon", "pipe", "plank", "pandp"]))],
                       synopsis='display cartoon for specified residues')
        register(command_name, desc, cartoon)
