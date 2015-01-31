# vim: set expandtab ts=4 sw=4:

from chimera.core import cli, atomspec


def bogus(session, args="no arguments"):
    session.logger.info("bogus: %s" % args)
bogus_desc = cli.CmdDesc(optional=[("args", cli.RestOfLine)])


def hidewater(session, modelspec=None):
    print("modelspec", modelspec)
    all_models = session.models.list()
    print("all_models", all_models)
    wanted_models = set()
    modelspec.evaluate(all_models, wanted_models)
    print("wanted_models", wanted_models)
    import numpy
    for m in wanted_models:
        atom_res_types = numpy.array(m.mol_blob.atoms.residues.names)
        indices = numpy.where(atom_res_types == "HOH")
        atom_draw_modes = m.mol_blob.atoms.draw_modes
        atom_draw_modes[indices] = m.BALL_STYLE
        ad = numpy.array(atom_draw_modes)
        m.mol_blob.atoms.draw_modes = ad
        # TODO: Parallel code does not work for atom_displays.
        # TODO: Can we get numpy arrays directly?
        m.update_graphics()
hidewater_desc = cli.CmdDesc(optional=[("modelspec", atomspec.AtomSpecArg)])
