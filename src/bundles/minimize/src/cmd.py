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

from chimerax.core.errors import UserError, LimitationError
from chimerax.add_charge import ChargeMethodArg

def cmd_minimize(session, structure, *, dock_prep=True, live_updates=True, log_energy=False,
        max_steps=None, **kw):
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
        dock_prep_caller(session, [structure], memorize_name="minimization", nogui=True,
            callback=lambda ses=session, struct=structure, updates=live_updates, log=log_energy,
            steps=max_steps: _minimize(ses, struct, updates, log, steps), **kw)
    else:
        _minimize(session, structure, live_updates, log_energy, max_steps)

def _minimize(session, structure, live_updates, log_energy, max_steps):
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
    # Also, it can't handle missing structure, so make ends of missing structure look like terminii
    fake_c = set()
    fake_n = set()
    for chain in structure.chains:
        in_missing = False
        prev_r = None
        for r in chain.residues:
            if r is None:
                if prev_r:
                    fake_c.add(prev_r)
                in_missing = True
            else:
                if in_missing:
                    fake_n.add(r)
                in_missing = False
            prev_r = r
    n_error_template = "Don't know how to modify %s to match N-terminal template: %s"
    c_error_template = "Don't know how to modify %s to match C-terminal template: %s"
    from chimerax.atomic.bond_geom import bond_positions
    from chimerax.addh import bond_with_H_length
    NH_len = CO_len = None
    filter = []
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
            atoms[a] = top.addAtom(a.name, element.Element.getBySymbol(a.element.name), mm_r,
                                    a.serial_number)
            coords.append(Quantity(vec3.Vec3(*a.coord), angstrom))
            reordered_atoms.append(a)
            filter.append(True)
        # add fake N/C-terminal atoms only to the OpenMM version of the molecule
        fake_serial = max(structure.atoms.serial_numbers) + 1
        if r in fake_n:
            n = r.find_atom('N')
            if not n:
                raise LimitationError(n_error_template % (r, "can't find N atom"))
            if NH_len is None:
                NH_len = bond_with_H_length(n, 3)
            bonded = n.neighbors
            if not bonded:
                raise LimitationError(n_error_template % (r, "N atom has no bonds"))
            for pos in bond_positions(n.coord, 4, NH_len, [bd.coord for bd in bonded]):
                num = 1
                while r.find_atom("H%d" % num):
                    num += 1
                h = top.addAtom("H%d" % num, element.Element.getBySymbol("H"), residues[r], fake_serial)
                coords.append(Quantity(vec3.Vec3(*pos), angstrom))
                top.addBond(atoms[n], h)
                fake_serial += 1
                filter.append(False)
        if r in fake_c:
            c = r.find_atom("C")
            if not c:
                raise LimitationError(c_error_template % (r, "can't find C atom"))
            if CO_len is None:
                from chimerax.atomic import Element
                CO_len = Element.bond_length("C", "O")
            bonded = c.neighbors
            if len(bonded) != 2:
                raise LimitationError(c_error_template % (r, "C atom not bonded to exactly two other atoms"))
            for pos in bond_positions(c.coord, 3, CO_len, [bd.coord for bd in bonded]):
                o = top.addAtom("OXT", element.Element.getBySymbol("O"), residues[r], fake_serial)
                coords.append(Quantity(vec3.Vec3(*pos), angstrom))
                top.addBond(atoms[c], o)
                fake_serial += 1
                filter.append(False)
    filter = numpy.array(filter)

    for b in structure.bonds:
        top.addBond(atoms[b.atoms[0]], atoms[b.atoms[1]])

    forcefield = ForceField('amber14-all.xml', 'amber14/tip3pfb.xml')
    unmatched_omm_residues = forcefield.getUnmatchedResidues(top)
    if unmatched_omm_residues:
        #from .parameterize import parameterize
        from chimerax.core.commands import commas
        print("Unmatched residues: %s" % commas([rname for rname in set([r.name for r in unmatched_omm_residues])], conjunction="and"))
        #TODO: need to find these residues in structure, sort into isomers, for each generate a
        # ForceField._TemplateData (see openmm.app.forcefield), possibly add a distinguishing isomer
        # number, and register template
        raise LimitationError("Non-standard residues in structure; see log for details")
    try:
        system = forcefield.createSystem(top, nonbondedCutoff=1*nanometer, constraints=HBonds)
    except ValueError as e:
        err_text = str(e)
        if err_text.startswith("No template"):
            left_paren = err_text.find('(')
            right_paren = err_text.find(')')
            if left_paren >= 0 and right_paren > left_paren:
                raise LimitationError("Support for minimizing structures with non-standard residues"
                    " (such as %s) not yet implemented.  If such residues are not crucial for your"
                    " analysis, consider deleting them and then minimizing."
                    % err_text[left_paren+1:right_paren])
        raise
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
                session.logger.status("step %d: energy %.1f" % (self.step, args[0]["system energy"]),
                    log=log_energy)
                if live_updates:
                    crds = numpy.array(xyz)
                    crds = numpy.reshape(crds, (-1,3))
                    self.atoms.coords = crds[filter] * 10
                    from chimerax.core.commands import run
                    run(session, "wait 1", log=False)
            return False if max_steps is None else self.step >= max_steps
    from chimerax.atomic import Atoms
    cx_atoms = Atoms(reordered_atoms)
    # maxIterations doesn't truly constrain maximum iterations as you would expect
    # (see https://github.com/openmm/openmm/issues/4983), so it is handled in the reporter instead
    LocalEnergyMinimizer.minimize(context, reporter=Reporter(cx_atoms))
    final_crds = numpy.array([q.value_in_unit(angstrom)
        for q in context.getState(getPositions=True).getPositions()])
    final_crds = numpy.reshape(final_crds, (-1,3))
    cx_atoms.coords = final_crds[filter]

def register_command(logger):
    from chimerax.core.commands import CmdDesc, register, Or, EmptyArg, EnumOf, BoolArg, PositiveIntArg
    from chimerax.atomic import AtomicStructureArg
    from chimerax.dock_prep import get_param_info
    desc = CmdDesc(
        required = [('structure', Or(AtomicStructureArg, EmptyArg))],
        keyword = [
            ('dock_prep', BoolArg),
            ('live_updates', BoolArg),
            ('log_energy', BoolArg),
            ('max_steps', PositiveIntArg),
        ] + list(get_param_info(logger.session).items()),
        synopsis = 'Minimize structures'
    )
    register("minimize", desc, cmd_minimize, logger=logger)
