# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2022 Regents of the University of California. All rights reserved.
# This software is provided pursuant to the ChimeraX license agreement, which
# covers academic and commercial uses. For more information, see
# <http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
#
# This file is part of the ChimeraX library. You can also redistribute and/or
# modify it under the GNU Lesser General Public License version 2.1 as
# published by the Free Software Foundation. For more details, see
# <https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html>
#
# This file is distributed WITHOUT ANY WARRANTY; without even the implied
# warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. This notice
# must be embedded in or attached to all copies, including partial copies, of
# the software or any revisions or derivations thereof.
# === UCSF ChimeraX Copyright ===

from chimerax.core.commands import EnumOf
ChargeMethodArg = EnumOf(['am1-bcc', 'gasteiger'])
ChargeMethodArg.default_value = 'am1-bcc'

from chimerax.core.errors import UserError
from .charge import default_standardized, add_charges, add_nonstandard_res_charges, ChargeError

# functions in .dock_prep may need updating if cmd_addcharge() call signature changes
def cmd_addcharge(session, residues, *, method=ChargeMethodArg.default_value,
        standardize_residues=default_standardized):
    if residues is None:
        from chimerax.atomic import all_residues
        residues = all_residues(session)
    if not residues:
        raise UserError("No residues specified")

    if standardize_residues == "none":
        standardize_residues = []
    check_hydrogens(session, residues)
    try:
        add_charges(session, residues, method=method, status=session.logger.status,
            standardize_residues=standardize_residues)
    except ChargeError as e:
        raise UserError(str(e))
    from math import floor
    def is_non_integral(val):
        return abs(floor(val+0.5) - val) > 0.005
    non_integral_info = {}
    for s, s_residues in residues.by_structure:
        non_integral = []
        total_charge = 0.0
        for r in s_residues:
            res_charge = sum([a.charge for a in r.atoms])
            if is_non_integral(res_charge):
                non_integral.append((r, res_charge))
            total_charge += res_charge
        if non_integral and is_non_integral(total_charge):
            non_integral_info[s] = non_integral
    if non_integral_info:
        from chimerax.core.commands import plural_form, commas
        session.logger.warning("%d %s has non-integral total charge.\n"
            "Details in log." % (len(non_integral_info), plural_form(non_integral_info, "structure")))
        session.logger.info("Here are the structures will non-integral total charge along with the"
            " particular residues from those structures with non-integral total charge:")
        for s, res_info in non_integral_info.items():
            if len(res_info) == len([r for r in s.residues if r.num_atoms > 1]):
                res_info_text = "all residues"
            else:
                res_info = commas(["%s: %g" % (r,c) for r, c in res_info], conjunction="and")
            session.logger.info(f"{s}: {res_info}")

def cmd_addcharge_nonstd(session, residues, res_name, net_charge, *,
        method=ChargeMethodArg.default_value):
    if residues is None:
        from chimerax.atomic import all_residues
        residues = all_residues(session)
    residues = residues.filter(residues.names == res_name)
    if not residues:
        raise UserError(f"No specified residues are named '{res_name}'")

    check_hydrogens(session, residues)
    add_nonstandard_res_charges(session, residues, net_charge, method=method, status=session.logger.status)

def check_hydrogens(session, residues):
    atoms = residues.atoms
    hyds = atoms.filter(atoms.element_numbers == 1)
    if hyds:
        return
    if session.in_script:
        return
    from chimerax.ui.ask import ask
    if ask(session, "Adding charges requires hydrogen atoms to be present.", show_icon=False,
            info="The residues requested have no hydrogen atoms.\n"
            "Add hydrogens to them now?") == "yes":
        from chimerax.core.commands import run, StringArg
        from chimerax.atomic import concise_residue_spec
        run(session, "addh %s" % StringArg.unparse(concise_residue_spec(session, residues)))

def register_command(logger):
    from chimerax.core.commands import CmdDesc, register, Or, EmptyArg, EnumOf, IntArg, StringArg, ListOf
    from chimerax.core.commands import NoneArg
    from chimerax.atomic import ResiduesArg
    from chimerax.atomic.struct_edit import standardizable_residues
    desc = CmdDesc(
        required = [('residues', Or(ResiduesArg, EmptyArg))],
        keyword = [
            ('method', ChargeMethodArg),
            ('standardize_residues', Or(ListOf(EnumOf(standardizable_residues)),NoneArg)),
        ],
        synopsis = 'Add charges'
    )
    register("addcharge", desc, cmd_addcharge, logger=logger)

    desc = CmdDesc(
        required = [('residues', Or(ResiduesArg, EmptyArg)), ('res_name', StringArg),
            ('net_charge', IntArg)],
        keyword = [
            ('method', ChargeMethodArg),
        ],
        synopsis = 'Add non-standard residue charges'
    )
    register("addcharge nonstd", desc, cmd_addcharge_nonstd, logger=logger)
