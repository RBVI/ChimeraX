# vi: set expandtab shiftwidth=4 softtabstop=4:


def cartoon(session, spec=None, adjust=None, style=None):
    '''Display cartoon for specified residues.

    Parameters
    ----------
    spec : atom specifier
        Show ribbons for the specified residues. If no atom specifier is given then ribbons are shown
        for all residues.  Residues that are already shown as ribbons remain shown as ribbons.
    adjust : floating point number
        Adjustment factor for strand and helix smoothing.  A factor of zero means the
        cartoon will pass through the atom position.  A factor of one means the cartoon
        will pass through the "ideal" position, e.g., center of the cylinder that best
        fits a helix.  A factor of "default" means to return to default (0.7 for strands
        and 0 for everything else).
    style : string
        Set "Ribbon" style.  Value may be "ribbon" for normal ribbons, or one of "pipe",
        "plank", or "pandp" to display residues as pipes and planks.
    '''
    if spec is None:
        from chimera.core.commands import atomspec
        spec = atomspec.everything(session)
    results = spec.evaluate(session)
    residues = results.atoms.residues
    residues.ribbon_displays = True
    if adjust is not None:
        if adjust is "default":
            # Convert to C++ default value
            adjust = -1.0
        residues.ribbon_adjusts = adjust
    if style is not None:
        # Convert to C++ value
        from atomic import Residue
        if style == "ribbon":
            s = Residue.RIBBON
        elif style in ["pipe", "plank", "pandp"]:
            s = Residue.PIPE
        residue.ribbon_styles = s


def uncartoon(session, spec=None):
    '''Undisplay ribbons for specified residues.

    Parameters
    ----------
    spec : atom specifier
        Hide ribbons for the specified residues. If no atom specifier is given then all ribbons are hidden.
    '''
    if spec is None:
        from chimera.core.commands import atomspec
        spec = atomspec.everything(session)
    results = spec.evaluate(session)
    results.atoms.residues.ribbon_displays = False


def initialize(command_name):
    from chimera.core.commands import register
    from chimera.core.commands import CmdDesc, AtomSpecArg, Or, Bounded, FloatArg, EnumOf
    desc = CmdDesc(optional=[("spec", AtomSpecArg),
                             ("adjust", Or(Bounded(FloatArg, 0.0, 1.0),
                                           EnumOf(["default"]))),
                             ("style", EnumOf(["ribbon", "pipe", "plank", "pandp"]))],
                   synopsis='display cartoon for specified residues')
    register(command_name, desc, cartoon)
    desc = CmdDesc(optional=[("spec", AtomSpecArg)],
                   synopsis='undisplay cartoon for specified residues')
    register("~" + command_name, desc, uncartoon)
