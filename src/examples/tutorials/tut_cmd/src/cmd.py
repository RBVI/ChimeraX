# vim: set expandtab shiftwidth=4 softtabstop=4:

from chimerax.core.commands import CmdDesc      # Command description
from chimerax.atomic import AtomsArg            # Collection of atoms argument
from chimerax.core.commands import BoolArg      # Boolean argument
from chimerax.core.commands import ColorArg     # Color argument
from chimerax.core.commands import IntArg       # Integer argument
from chimerax.core.commands import EmptyArg     # (see below)
from chimerax.core.commands import Or, Bounded  # Argument modifiers


# ==========================================================================
# Functions and descriptions for registering using ChimeraX bundle API
# ==========================================================================


def cofm(session, atoms, *, weighted=False, transformed=True):
    """Report center of mass of given atoms."""

    # ``session``     - ``chimerax.core.session.Session`` instance
    # ``atoms``       - ``chimerax.atomic.Atoms`` instance or None
    # ``weighted``    - boolean, whether to include atomic mass in calculation
    # ``transformed`` - boolean, use scene rather than original coordinates

    atoms, coords, cofm = _get_cofm(session, atoms, transformed, weighted)
    session.logger.info("%s center of mass: %s" %
                        ("weighted" if weighted else "unweighted", cofm))


cofm_desc = CmdDesc(required=[("atoms", Or(AtomsArg, EmptyArg))],
                    keyword=[("weighted", BoolArg),
                              ("transformed", BoolArg)])

# CmdDesc contains a description of how the user-typed command arguments
# should be translated into Python arguments for the function that actually
# implements the command.  There are three styles:
#   required:  the user must provide such arguments and the Python function
#       should declare them as mandatory (i.e. with no default value).
#   optional:  the user can optionally provide these arguments immediately
#       after the mandatory arguments.  The Python function should provide
#       default values for them.
#   keyword:  the user must provide a keyword to specify these arguments, but
#       they can be in any order after the required and optional arguments.
#       The Python function normally declares them after a '*,' (which
#       indicates the start of Python keyword-only arguments)
#
# Most commands should only use required and keyword arguments.  Optional
# arguments should be used in the rare case that an argument's meaning is
# obvious from its position, and it is acceptable for the argument to be
# missing.  For instance, the ChimeraX 'torsion' command requires an atom
# spec specifying four atoms as its first argument, but accepts an optional
# floating point value as a second argument.  If only the first argumeht is
# given, the torsion angle value is reported.  If both arguments are given,
# then the torsion angle is set to that value.
#
# The required/optional/keyword descriptions are passed as keyword arguments
# to the ``CmdDesc`` constructor.  Each set of descriptions is passed as a
# list of 2-tuples.  The first element of the tuple must match the name of a
# parameter of the callback function.  The second element must be a class
# describing the expected input; ChimeraX provides many such classes, e.g.,
# BoolArg for boolean values, IntArg for integer values, etc.  The order of
# the 2-tuples in the required and optional lists determine the order that
# the user must provide those arguments to the command.  That need not be the
# same order as the arguments of the callback function, though it is typically
# good programming practice to have the orders the same.

# For the "cofm" command, we declare three arguments:
#   ``atoms``       - collection of atoms (required), default: all atoms
#   ``weighted``    - boolean (keyword), default: False
#   ``transformed`` - boolean (keyword), default: True
#
# Example commands:
#   tut cofm /A                (cofm of chain A)
#   tut cofm weighted t        (weighted cofm of all atoms)
#   tut cofm :23 trans false   (cofm of input coordinates of residue 23)
#
# Note the trick used for the "atoms" argument, which may be left out to
# mean "use all atoms".  If we make "atoms" a keyword argument, the user
# would have to enter "tut cofm atoms /A" rather than "tut cofm /A".
# The trick is to make "atoms" required, so that the typed command does not
# need to include the "atoms" keyword; the value for "atoms" can be either
# an AtomsArg or an EmptyArg.  If the user enters an atom specification as
# part of the command, then "atoms" value matches AtomsArg, which
# translates to a ``chimerax.atomic.Atoms`` instance for the function
# parameter; if not, "atoms" matches EmptyArg, which translates to ``None``.
#
# The astute reader will note that the "atoms" argument could have instead 
# been declared as optional, with a parser class of AtomsArg and a Python
# default value of None.  In ChimeraX, commands that require some kind of
# atom specification typically have that as their first argument.  If the
# command has additional required arguments, then you would have to use the
# "trick" demonstrated here in order to allow the atom spec to be the first
# argument while still allowing that argument to be omitted in order to
# indicate "all atoms".


def highlight(session, atoms, color, *, weighted=False, transformed=True, count=1):
    """Highlight the atoms nearest the center of mass of given atoms."""

    # ``session``     - ``chimerax.core.session.Session`` instance
    # ``atoms``       - ``chimerax.atomic.Atoms`` instance or None
    # ``color``       - ``chimerax.core.colors.Color` instance
    # ``weighted``    - boolean, whether to include atomic mass in calculation
    # ``transformed`` - boolean, use scene rather than original coordinates

    # Compute the center of mass first
    atoms, coords, cofm = _get_cofm(session, atoms, transformed, weighted)

    # Compute the distance of each atom from the cofm
    # using the NumPy vector norm function
    from numpy.linalg import norm
    distances = norm(coords - cofm, axis=1)

    # Sort the array and get the "count" indices to the closest atoms
    if count > len(atoms):
        count = len(atoms)
    from numpy import argsort
    atom_indices = argsort(distances)[:count]

    # Create a collection of atoms from the indices
    chosen = atoms[atom_indices]

    # Update their "colors".  Assigning a single value to an
    # array means assign the same value for all elements.
    chosen.colors = color.uint8x4()


highlight_desc = CmdDesc(required=[("atoms", Or(AtomsArg, EmptyArg)),
                                   ("color", ColorArg)],
                         keyword=[("weighted", BoolArg),
                                   ("transformed", BoolArg),
                                   ("count", Bounded(IntArg, 1, 5))])


# ==========================================================================
# Functions intended only for internal use by bundle
# ==========================================================================


def _get_cofm(session, atoms, transformed, weighted):
    # ``session``     - ``chimerax.core.session.Session`` instance
    # ``atoms``       - ``chimerax.atomic.Atoms`` instance
    # ``transformed`` - boolean, use scene rather than original coordinates
    # ``weighted``    - boolean, whether to include atomic mass in calculation

    # If user did not specify the list of atoms, use all atoms
    if atoms is None:
        from chimerax.atomic import all_atoms
        atoms = all_atoms(session)

    # We can use either transformed or untransformed coordinates.
    # Transformed coordinates are "scene coordinates", which
    # takes into account translation and rotation of individual
    # models.  Untransformed coordinates are the coordinates
    # read from the data files.
    if transformed:
        coords = atoms.scene_coords
    else:
        coords = atoms.coords

    # ``coords`` is a ``numpy`` float array of shape (N, 3)
    # If we want weighted center, we have to multiply coordinates
    # by the atomic mass
    if not weighted:
        cofm = coords.mean(axis=0)
    else:
        m = atoms.elements.masses
        c = coords * m[:,None]
        cofm = c.sum(axis=0) / m.sum()

    # To get the average coordinates, we use ``numpy.mean``
    # print("DEBUG: center of mass:", cofm)
    return atoms, coords, cofm
