# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2022 Regents of the University of California. All rights reserved.
# The ChimeraX application is provided pursuant to the ChimeraX license
# agreement, which covers academic and commercial uses. For more details, see
# <https://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
#
# This particular file is part of the ChimeraX library. You can also
# redistribute and/or modify it under the terms of the GNU Lesser General
# Public License version 2.1 as published by the Free Software Foundation.
# For more details, see
# <https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html>
#
# THIS SOFTWARE IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND, EITHER
# EXPRESSED OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
# OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. ADDITIONAL LIABILITY
# LIMITATIONS ARE DESCRIBED IN THE GNU LESSER GENERAL PUBLIC LICENSE
# VERSION 2.1
#
# This notice must be embedded in or attached to all copies, including partial
# copies, of the software or any revisions or derivations thereof.
# === UCSF ChimeraX Copyright ===

from chimerax.core.errors import UserError
from chimerax.add_charge import ChargeMethodArg

def cmd_minimize(session, structure, *, dock_prep=True, **kw):
    if structure is None:
        from chimerax.atomic import all_atomic_structures
        available = all_atomic_structures(session)
        if len(available) == 0:
            raise UserError("No structures open")
        elif len(available) == 1:
            structure = available[0]
        else:
            raise UserError("Multiple structures open")
    if dock_prep:
        from chimerax.dock_prep import dock_prep_caller
        dock_prep_caller(session, [structure], memorize_name="minimization",
            callback=lambda ses=session, struct=structure: _minimize(ses, struct), **kw)
    else:
        _minimize(session, structure)

def _minimize(session, structure):
    from openmm.app import Topology, ForceField, element, HBonds
    from openmm.unit import angstrom, nanometer, kelvin, picosecond, picoseconds, Quantity
    from openmm import LangevinIntegrator, LocalEnergyMinimizer, vec3, Context, MinimizationReporter
    import numpy
    top = Topology()
    atoms = {}
    residues= {}
    chains = {}
    coords = []
    reordered_atoms = []
    # OpenMM wants all atoms in a residue to be added consecutively, so loop through residues instead of atoms...
    for r in structure.residues:
        c = r.chain
        try:
            mm_c = chains[c]
        except KeyError:
            mm_c = chains[c] = top.addChain("singletons" if c is None else c)
        try:
            mm_r = residues[r]
        except KeyError:
            mm_r = residues[r] = top.addResidue(r.name, mm_c, r.number, r.insertion_code)
        for a in r.atoms:
            atoms[a] = top.addAtom(a.name, element.Element.getBySymbol(a.element.name), mm_r, a.serial_number)
            coords.append(Quantity(vec3.Vec3(*a.coord), angstrom))
            reordered_atoms.append(a)

    for b in structure.bonds:
        top.addBond(atoms[b.atoms[0]], atoms[b.atoms[1]])

    forcefield = ForceField('amber14-all.xml', 'amber14/tip3pfb.xml')
    system = forcefield.createSystem(top, nonbondedCutoff=1*nanometer, constraints=HBonds)
    integrator = LangevinIntegrator(300*kelvin, 1/picosecond, 0.004*picoseconds)
    context = Context(system, integrator)
    context.setPositions(Quantity(coords))
    class Reporter(MinimizationReporter):
        step = 0
        report_interval = 100

        def __init__(self, atoms):
            self.atoms = atoms
            super().__init__()

        def report(self, iteration, xyz, gradient, *args):
            self.step += 1
            if self.step % self.report_interval == 0:
                session.logger.status("step %d: energy %.1f" % (self.step, args[0]["system energy"]), log=True)
                crds = numpy.array(xyz)
                crds = numpy.reshape(crds, (-1,3))
                self.atoms.coords = crds * 10
                from chimerax.core.commands import run
                run(session, "wait 1", log=False)
            return False
    from chimerax.atomic import Atoms
    cx_atoms = Atoms(reordered_atoms)
    LocalEnergyMinimizer.minimize(context, reporter=Reporter(cx_atoms))
    final_crds = numpy.array([q.value_in_unit(angstrom)
        for q in context.getState(getPositions=True).getPositions()])
    final_crds = numpy.reshape(final_crds, (-1,3))
    cx_atoms.coords = final_crds

def register_command(logger):
    from chimerax.core.commands import CmdDesc, register, Or, EmptyArg, EnumOf, BoolArg
    from chimerax.atomic import AtomicStructureArg
    from chimerax.dock_prep import get_param_info
    desc = CmdDesc(
        required = [('structure', Or(AtomicStructureArg, EmptyArg))],
        keyword = [
            ('dock_prep', BoolArg),
        ] + list(get_param_info(logger.session).items()),
        synopsis = 'Minimize structures'
    )
    register("minimize", desc, cmd_minimize, logger=logger)
