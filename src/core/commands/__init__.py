# === UCSF ChimeraX Copyright ===
# Copyright 2016 Regents of the University of California.
# All rights reserved.  This software provided pursuant to a
# license agreement containing restrictions on its disclosure,
# duplication and use.  For details see:
# http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html
# This notice must be embedded in or attached to all copies,
# including partial copies, of the software or any revisions
# or derivations thereof.
# === UCSF ChimeraX Copyright ===

from .run import run, concise_model_spec
from .runscript import runscript

from .cli import CmdDesc, register, Command, create_alias, command_function
from .cli import commas, plural_form, plural_of, discard_article
from .cli import ListOf, SetOf, TupleOf, Or, RepeatOf

from .cli import Annotation, AnnotationError, next_token, quote_if_necessary
from .cli import NoArg, NoneArg, BoolArg, StringArg, EmptyArg, EnumOf, DynamicEnum
from .cli import IntArg, Int2Arg, Int3Arg, NonNegativeIntArg, PositiveIntArg
from .cli import FloatArg, Float2Arg, Float3Arg, FloatsArg
from .cli import AxisArg, Axis, CenterArg, Center, CoordSysArg, PlaceArg, Bounded
from .cli import ModelIdArg, AtomsArg, UniqueChainsArg, AtomicStructuresArg
from .cli import StructureArg, StructuresArg, SurfacesArg, SurfaceArg
from .cli import BondArg, BondsArg, PseudobondsArg, PseudobondGroupsArg
from .cli import ModelArg, ModelsArg, TopModelsArg, ObjectsArg, RestOfLine
from .cli import OpenFileNameArg, SaveFileNameArg, OpenFolderNameArg, SaveFolderNameArg
from .cli import AttrNameArg

from .colorarg import ColorArg, Color8Arg, ColormapArg, ColormapRangeArg
from .symarg import SymmetryArg

from .atomspec import AtomSpecArg, all_objects
from .atomspec import register_selector, deregister_selector
from .atomspec import list_selectors, get_selector, get_selector_description
from .atomspec import is_selector_user_defined, is_selector_atomic
