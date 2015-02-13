# vim: set expandtab ts=4 sw=4:

from chimera.core import cli, atomspec
from chimera.core.webservices.opal_job import OpalJob


def echo(session, args="no arguments"):
    session.logger.info("echo: %s" % args)
echo_desc = cli.CmdDesc(optional=[("args", cli.RestOfLine)])


def hidewater(session, modelspec=None):
    spec = modelspec.evaluate(session)
    import numpy
    for m in spec.models:
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
    spec = modelspec.evaluate(session)
    import numpy
    by_vector = numpy.array(by)
    from chimera.core.geometry import place
    translation = place.translation(by_vector)
    for m in spec.models:
        m.position = translation * m.position
        m.update_graphics()
move_desc = cli.CmdDesc(required=[("by", cli.Float3Arg)],
                        optional=[("modelspec", atomspec.AtomSpecArg)])


class CCD(OpalJob):
    def __init__(self, session, name):
        super().__init__(session)
        self.start("CCDService", name)

    def on_finish(self):
        print("Standard output:")
        print(self.get_file("stdout.txt").decode(encoding="UTF-8"))


def opal(session, name):
    CCD(session, name)
opal_desc = cli.CmdDesc(required=[("name", cli.StringArg)])
