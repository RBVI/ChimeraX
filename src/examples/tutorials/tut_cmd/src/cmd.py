# vim: set expandtab shiftwidth=4 softtabstop=4:

from chimerax.core.commands import CmdDesc      # Command description
from chimerax.core.commands import AtomsArg     # Collection of atoms argument
from chimerax.core.commands import BoolArg      # Boolean argument
from chimerax.core.commands import ColorArg     # Color argument
from chimerax.core.commands import IntArg       # Integer argument
from chimerax.core.commands import EmptyArg     # (see below)
from chimerax.core.commands import Or, Bounded  # Argument modifiers


# ==========================================================================
# Functions and descriptions for registering using ChimeraX bundle API
# ==========================================================================


def cofm(session, atoms, weighted=False, transformed=True):
    """Report center of mass of given atoms."""

    # ``session``     - ``chimerax.core.session.Session`` instance
    # ``atoms``       - ``chimerax.core.atomic.Atoms`` instance or None
    # ``weighted``    - boolean, whether to include atomic mass in calculation
    # ``transformed`` - boolean, use scene rather than original coordinates

    atoms, coords, cofm = _get_cofm(session, atoms, transformed, weighted)
    session.logger.info("%s center of mass: %s" %
                        ("weighted" if weighted else "unweighted", cofm))

# CmdDesc contains the command description.
# For the "cofm" command, we expect three arguments:
#   ``atoms``       - collection of atoms (required), default: all atoms
#   ``weighted``    - boolean (optional), default: False
#   ``transformed`` - boolean (optional), default: True
# ChimeraX expects the command syntax to be something like:
#   command_name req1 req2 [opt1_keyword opt1 value] [opt2_keyword opt2_value]
# where reqX is the value for a required argument, and optX_keyword and
# optX_value are the keyword and value for an optional argument.
# Required arguments are listed in the order expected.  Optional arguments
# appear after required arguments but may be in any order.
#
# Example commands:
#   tut cofm /A                (cofm of chain A)
#   tut cofm weighted t        (weighted cofm of all atoms)
#   tut cofm :23 trans false   (cofm of input coordinates of residue 23)
#
# The required and optional arguments are passed as keyword arguments to
# the ``CmdDesc`` constructor.  Each set of arguments is passed as a
# list of 2-tuples.  The first element of the tuple must match the name
# of a parameter of callback function.  The second element must be a class
# describing the expected input; ChimeraX provides many such classes,
# e.g., BoolArg for boolean values, IntArg for integer values, etc.
# The order of tuples is important for require arguments as the user
# must enter them in that order.  The order is irrelevant for optional
# arguments since they are identified by keywords.
#
# Note the trick used for the "atoms" argument, which may be left out to
# mean "use all atoms".  If we make "atoms" an optional argument, the user
# would have to enter "tut cofm atoms /A" rather than "tut cofm /A".
# The trick is to make "atoms" required, so the input does not need to
# include the "atoms" keyword; the value for "atoms" can be either an
# AtomsArg or an EmptyArg.  If the user enters an atom specification as
# part of the command, then "atoms" value matches AtomsArg, which
# translates to a ``chimerax.core.atomic.Atoms`` instance for the function
# parameter; if not, "atoms" matches EmptyArg, which translates to ``None``.
#
cofm_desc = CmdDesc(required=[("atoms", Or(AtomsArg, EmptyArg))],
                    optional=[("weighted", BoolArg),
                              ("transformed", BoolArg)])


def highlight(session, atoms, color, weighted=False, transformed=True, count=1):
    """Highlight the atoms nearest the center of mass of given atoms."""

    # ``session``     - ``chimerax.core.session.Session`` instance
    # ``atoms``       - ``chimerax.core.atomic.Atoms`` instance or None
    # ``color``       - ``chimerax.core.colors.Color` instance
    # ``weighted``    - boolean, whether to include atomic mass in calculation
    # ``transformed`` - boolean, use scene rather than original coordinates

    # Compute the center of mass first
    atoms, coords, cofm = _get_cofm(session, atoms, transformed, weighted)

    # Compute the distance of each atom from the cofm
    from numpy.linalg import norm
    distances = norm(coords - cofm, axis=1)

    # Sort the array so to get the indices to the closest atoms
    if count > len(atoms):
        count = len(atoms)
    from numpy import argsort
    atom_indices = argsort(distances)[:count]

    # Collect the atoms we need
    chosen = atoms[atom_indices]

    # Update their colors
    chosen.colors = color.uint8x4()


highlight_desc = CmdDesc(required=[("atoms", Or(AtomsArg, EmptyArg)),
                                   ("color", ColorArg)],
                         optional=[("weighted", BoolArg),
                                   ("transformed", BoolArg),
                                   ("count", Bounded(IntArg, 1, 5))])


# ==========================================================================
# Functions intended only for internal use by bundle
# ==========================================================================


def _get_cofm(session, atoms, transformed, weighted):
    # ``session``     - ``chimerax.core.session.Session`` instance
    # ``atoms``       - ``chimerax.core.atomic.Atoms`` instance
    # ``transformed`` - boolean, use scene rather than original coordinates
    # ``weighted``    - boolean, whether to include atomic mass in calculation

    # If user did not specify the list of atoms, use all atoms
    if atoms is None:
        from chimerax.core.commands import all_objects
        atoms = all_objects(session).atoms

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
    if weighted:
        coords *= atoms.elements.masses

    # To get the average coordinates, we use ``numpy.mean``
    import numpy
    cofm = numpy.mean(coords, axis=0)
    # print("DEBUG: center of mass:", cofm)
    return atoms, coords, cofm
