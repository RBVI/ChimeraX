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

from .idatm import type_info

def chirality(atom):
    try:
        info = type_info[atom.idatm_type]
    except KeyError:
        if atom.num_bonds != 4:
            return None
    else:
        if info.substituents != 4:
            return None

    try:
        center_branch_atoms = [BranchAtom(nb) for nb in atom.neighbors]
        branches = [Branch(cba) for cba in center_branch_atoms]
        visited = set(center_branch_atoms)
        visited.add(BranchAtom(atom))
        if len(branches) < 3:
            return None
        elif len(branches) < 4:
            branches.append(Branch(phantom_H))
        if branches_distinct(branches, visited):
            branches.sort(key=lambda br: -br.priority)
            from chimerax.geometry import dihedral
            coords = [branches[0].root.atom.coord, atom.coord] + [br.root.atom.coord
                for br in branches[1:3]]
            return 'R' if dihedral(*coords) > 0 else 'S'
        else:
            return None
    finally:
        BranchAtom.a_to_branch_a.clear()

def branches_distinct(branches, visited):
    # see if the current branch wave resolves everything
    unresolved_branches = [br for br in branches if not isinstance(br.priority, int)]
    # multiple empty branches -> non-chiral
    if [bool(ub.wave) for ub in unresolved_branches].count(False) > 1:
        return False
    import operator
    while True:
        if not unresolved_branches:
            return True
        if _find_unique_branch(unresolved_branches, min, operator.lt):
            continue
        if not _find_unique_branch(unresolved_branches, max, operator.gt):
            break

    # Add a wave and recurse.
    # Also expand along resolved branches so that unresolved branches don't "claim" atoms
    # that actually belong to a different branch
    new_additions = []
    for br in branches:
        new_wave = []
        for wa in br.wave:
            try:
                wa_info = type_info[wa.idatm_type]
            except KeyError:
                check_double = check_triple = False
            else:
                check_double = wa_info.geometry == 3 and not wa.already_double
                check_triple = wa_info.geometry == 2
            check_double = wa_info.geometry == 3 and not wa.already_double
            for nb in wa.neighbors:
                try:
                    bnb = BranchAtom.a_to_branch_a[nb]
                except KeyError:
                    bnb = BranchAtom(nb)
                    new_additions.append(bnb)
                if bnb in visited:
                    continue
                new_wave.append(bnb)

                # multiple bonds
                if check_double and type_info.get(bnb.idatm_type, None) == 3 and not bnb.already_double:
                    new_wave.append(bnb)
                    bnb.already_double = True
                if check_triple and type_info.get(bnb.idatm_type, None) == 2:
                    new_wave.append(bnb)

                # phantom atoms
                try:
                    subs = type_info[wa.idatm_type].substituents
                except KeyError:
                    subs = 0
                if subs > wa.num_bonds:
                    new_wave.extend([phantom_H] * (subs - wa.num_bonds))

        # sort wave
        new_wave.sort(key=lambda ba: -ba.atomic_number)
        br.wave = new_wave

    visited.update(new_additions)
    return branches_distinct(branches, visited)

def _find_unique_branch(unresolved_branches, min_max, operator):
        # Try to find unique priority...
        target_priority = min_max(sum([br.priority for br in unresolved_branches], start=[]))
        target_branches = [br for br in unresolved_branches if target_priority in br.priority]
        for i in range(max([len(tb.wave) for tb in target_branches])):
            target_number = None
            for tb in target_branches:
                try:
                    number = tb.wave[i].atomic_number
                except IndexError:
                    # shorter than other branches, lower priority..
                    number = 0
                if target_number is None or operator(number, target_number):
                    target_number = number
                    target_branches = [tb]
                elif target_number == number:
                    target_branches.append(tb)
            if len(target_branches) == 1:
                break
        if len(target_branches) > 1:
            return False
        target_branch = target_branches[0]
        target_branch.priority = target_priority
        unresolved_branches.remove(target_branch)
        for ub in unresolved_branches:
            if target_priority in ub.priority:
                ub.priority.remove(target_priority)
        if len(unresolved_branches) == 1:
            ub = unresolved_branches[0]
            ub.priority = ub.priority[0]
            unresolved_branches.clear()
        return True

import abc
class BaseAtom:
    @property
    @abc.abstractmethod
    def atomic_number(self):
        pass

    @property
    @abc.abstractmethod
    def idatm_type(self):
        pass

    @property
    @abc.abstractmethod
    def neighbors(self):
        pass

    @property
    @abc.abstractmethod
    def num_bonds(self):
        pass

class BranchAtom(BaseAtom):
    a_to_branch_a = {}

    def __init__(self, atom):
        self.atom = atom
        self.terminal = False
        self.a_to_branch_a[atom] = self
        self.already_double = False

    @property
    def atomic_number(self):
        return self.atom.element.number

    @property
    def idatm_type(self):
        return self.atom.idatm_type

    @property
    def neighbors(self):
        return self.atom.neighbors

    @property
    def num_bonds(self):
        return self.atom.num_bonds

class PhantomH(BaseAtom):
    @property
    def atomic_number(self):
        return 1

    @property
    def idatm_type(self):
        return 'H'

    @property
    def neighbors(self):
        return []

    @property
    def num_bonds(self):
        return 1

phantom_H = PhantomH()

class Branch:
    def __init__(self, root):
        self.root = root
        self.wave = [root]
        self.priority = [1,2,3,4]
