# vim: set expandtab ts=4 sw=4:

from chimera.core import cli, atomspec


def echo(session, args="no arguments"):
    session.logger.info("echo: %s" % args)
echo_desc = cli.CmdDesc(optional=[("args", cli.RestOfLine)])


def hidewater(session, modelspec=None):
    all_models = session.models.list()
    wanted_models = set()
    modelspec.evaluate(all_models, wanted_models)
    import numpy
    for m in wanted_models:
        atom_res_types = numpy.array(m.mol_blob.atoms.residues.names)
        indices = numpy.where(atom_res_types == "HOH")
        if True:
            atom_draw_modes = m.mol_blob.atoms.draw_modes
            atom_draw_modes[indices] = m.BALL_STYLE
            ad = numpy.array(atom_draw_modes)
            m.mol_blob.atoms.draw_modes = ad
        if True:
            atom_displays = m.mol_blob.atoms.displays
            atom_displays[indices] = False
            m.mol_blob.atoms.displays = atom_displays
        m.update_graphics()
hidewater_desc = cli.CmdDesc(optional=[("modelspec", atomspec.AtomSpecArg)])


def move(session, by, modelspec=None):
    all_models = session.models.list()
    wanted_models = set()
    modelspec.evaluate(all_models, wanted_models)
    import numpy
    by_vector = numpy.array(by)
    from chimera.core.geometry import place
    translation = place.translation(by_vector)
    for m in wanted_models:
        m.position = m.position * translation
        m.update_graphics()
move_desc = cli.CmdDesc(required=[("by", cli.Float3Arg)],
                        optional=[("modelspec", atomspec.AtomSpecArg)])
