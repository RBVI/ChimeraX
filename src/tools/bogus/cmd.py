# vi: set expandtab ts=4 sw=4:

from chimera.core import cli, atomspec
from chimera.core.webservices.opal_job import OpalJob
from .psize import Psize


def echo(session, args="no arguments"):
    session.logger.info("echo: %s" % args)
echo_desc = cli.CmdDesc(optional=[("args", cli.RestOfLine)])


def undisplay(session, spec=None):
    r = spec.evaluate(session)
    r.atoms.displays = False
    for m in r.models:
        m.update_graphics()
undisplay_desc = cli.CmdDesc(optional=[("spec", atomspec.AtomSpecArg)])


def display(session, spec=None):
    r = spec.evaluate(session)
    r.atoms.displays = True
    for m in r.models:
        m.update_graphics()
display_desc = cli.CmdDesc(optional=[("spec", atomspec.AtomSpecArg)])


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


def apbs(session, modelspec=None):
    if modelspec is None:
        models = session.models.list()
    else:
        models = modelspec.evaluate(session).models
    if len(models) != 1:
        raise UserError("apbs works on one model at a time")
    for m in models:
        _apbs_pdb2pqr(session, m)
apbs_desc = cli.CmdDesc(optional=[("modelspec", atomspec.AtomSpecArg)])


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


class _CCD(OpalJob):
    def __init__(self, session, name):
        super().__init__(session)
        self.start("CCDService", name)

    def on_finish(self):
        print("Standard output:")
        print(self.get_file("stdout.txt").decode(encoding="UTF-8"))


def opal(session, name):
    _CCD(session, name)
opal_desc = cli.CmdDesc(required=[("name", cli.StringArg)])


_ATOM_FMT = ("ATOM  %5d %-4s%1s"                # serial, atom name, altloc
             "%-3s %1s%4s%1s   "                # res name, chain, seq, insert
             "%8.3f%8.3f%8.3f%6.2f%6.2f      "  # xyz, occupancy, bfactor
             "%4s%2s%2s")                       # segment, element, charge
def _write_pdb(m, filename):
    atoms = m.mol_blob.atoms
    coords = atoms.coords
    atom_names = atoms.names
    element_names = atoms.element_names
    residues = atoms.residues
    chain_ids = residues.chain_ids
    residue_names = residues.names
    residue_numbers = residues.numbers
    with open(filename, "w") as f:
        serial = 0
        for i in range(len(coords)):
            serial += 1
            atom_name = atom_names[i]
            alt_loc = ' '
            res_name = residue_names[i]
            chain_id = chain_ids[i]
            res_seq = residue_numbers[i]
            res_insert_code = ' '
            x, y, z = coords[i]
            occupancy = 1.0
            b_factor = 0.0
            segment = ' '
            element = element_names[i]
            if len(element) == 1:
                atom_name = ' ' + atom_name
            charge = ' '
            print(_ATOM_FMT % (serial, atom_name, alt_loc,
                               res_name, chain_id, res_seq, res_insert_code,
                               x, y, z, occupancy, b_factor,
                               segment, element, charge),
                               file=f)


class _apbs_pdb2pqr(OpalJob):

    PDB_NAME = "apbs.pdb"
    PQR_NAME = "apbs.pqr"
    DX_NAME = "apbs.dx"
    OPAL_SERVICE = "pdb2pqr_1.9.0"
    OPAL_URL = "http://nbcr-222.ucsd.edu/opal2/services/"

    def __init__(self, session, m):
        super().__init__(session)
        import weakref
        self.mol = weakref.ref(m)
        _write_pdb(m, self.PDB_NAME)
        options = ["--chain",
                   "--ff", "amber",
                   self.PDB_NAME, self.PQR_NAME]
        cmd = ' '.join(options)
        input_file_map = [(self.PDB_NAME, "file", self.PDB_NAME)]
        self.start(self.OPAL_SERVICE, cmd, opal_url=self.OPAL_URL,
                   input_file_map=input_file_map)

    def on_finish(self):
        logger = self.session.logger
        logger.info("PDB2PQR standard output:")
        logger.info(self.get_file("stdout.txt").decode(encoding="UTF-8"))
        try:
            pqr = self.get_file(self.PQR_NAME)
        except KeyError:
            logger.error("pdb2pqr failed.")
            logger.error("PDB2PQR standard error:")
            logger.error(self.get_file("stderr.txt").decode(encoding="UTF-8"))
        else:
            with open(self.PQR_NAME, "w") as f:
                f.write(pqr.decode(encoding="UTF-8"))
            m = self.mol()
            if m is None:
                logger.error("molecule closed between pdb2pqr and apbs.")
            else:
                _apbs_apbs(self.session, m, self.PQR_NAME, self.DX_NAME)


class _apbs_apbs(OpalJob):

    CONFIG = "apbs.in"
    PREFIX = "apbs"
    SUFFIX = ".dx"
    OPAL_SERVICE = "apbs_1.3"
    OPAL_URL = "http://nbcr-222.ucsd.edu/opal2/services/"

    def __init__(self, session, m, pqr_name, dx_name):
        super().__init__(session)
        self.dx_name = dx_name
        cmd = [self.CONFIG]
        self._make_config(m, pqr_name)
        input_file_map = [(pqr_name, "file", pqr_name),
                          (self.CONFIG, "file", self.CONFIG)]
        self.start(self.OPAL_SERVICE, cmd, opal_url=self.OPAL_URL,
                   input_file_map=input_file_map)

    def _make_config(self, m, pqr_name):
        psize = MyPsize(m)
        with open(self.CONFIG, "w") as f:
            print("read", file=f)
            print("\tmol pqr %s" % pqr_name, file=f)
            print("end", file=f)

            print("elec", file=f)
            print("\tmg-auto", file=f)
            dime = psize.getFineGridPoints()
            print("\tdime %d %d %d" % (dime[0], dime[1], dime[2]), file=f)
            cglen = psize.getCoarseGridDims()
            print("\tcglen %.2f %.2f %.2f" % (cglen[0], cglen[1], cglen[2]),
                  file=f)
            fglen = psize.getFineGridDims()
            print("\tfglen %.2f %.2f %.2f" % (fglen[0], fglen[1], fglen[2]),
                  file=f)
            print("\tcgcent mol 1", file=f)
            print("\tfgcent mol 1", file=f)
            print("\tmol 1", file=f)
            print("\tlpbe", file=f)
            print("\tbcfl sdh", file=f)
            print("\tpdie 2.00", file=f)
            print("\tsdie 78.54", file=f)
            #print("\tsdie 2.00", file=f)
            print("\tchgm spl2", file=f)
            print("\tsrfm smol", file=f)
            print("\tswin 0.3", file=f)
            print("\tsdens 10.00", file=f)
            print("\tsrad 1.40", file=f)
            print("\ttemp 298.15", file=f)
            print("\tcalcenergy total", file=f)
            print("\tcalcforce no", file=f)
            print("\twrite pot dx %s" % self.PREFIX, file=f)
            print("end", file=f)

            print("quit", file=f)

    def on_finish(self):
        logger = self.session.logger
        logger.info("APBS standard output:")
        logger.info(self.get_file("stdout.txt").decode(encoding="UTF-8"))
        for dx in self.get_outputs().keys():
            if dx.startswith(self.PREFIX) and dx.endswith(self.SUFFIX):
                break
        else:
            logger.error("no .dx file generated.")
            logger.error("APBS standard error:")
            logger.error(self.get_file("stderr.txt").decode(encoding="UTF-8"))
            return
        try:
            dx = self.get_file(dx)
        except KeyError:
            logger.error("apbs failed.")
            logger.error("APBS standard error:")
            logger.error(self.get_file("stderr.txt").decode(encoding="UTF-8"))
        else:
            with open(self.dx_name, "wb") as f:
                f.write(dx)
            # TODO: open self.dx_name


#
# Derived class for estimating good initial parameter values
#
class MyPsize(Psize):

    def __init__(self, m):
        Psize.__init__(self)
        self._parse_molecule(m)
        self.setAll()

    def _parse_molecule(self, m):
        atoms = m.mol_blob.atoms
        coords = atoms.coords
        radii = atoms.radii
        num_atoms = len(coords)
        self.gotatom = len(coords)
        for i in range(num_atoms):
            rad = radii[i]
            center = coords[i]
            for i in range(3):
                lo = center[i] - rad
                if (self.minlen[i] == None or lo < self.minlen[i]):
                    self.minlen[i] = lo
                hi = center[i] + rad
                if (self.maxlen[i] == None or hi > self.maxlen[i]):
                    self.maxlen[i] = hi
