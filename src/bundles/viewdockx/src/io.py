# vim: set expandtab shiftwidth=4 softtabstop=4:

def open_mol2(session, path, file_name, auto_style, atomic):
    from chimerax.io import open_input
    with open_input(path, encoding='utf-8') as stream:
        p = Mol2Parser(session, stream, file_name, auto_style, atomic)
    structures = p.structures
    status = "Opened %s containing %d structures (%d atoms, %d bonds)" % (
                    file_name, len(structures),
                    sum([s.num_atoms for s in structures]),
                    sum([s.num_bonds for s in structures]))
    return structures, status


#
# Data types for mol2 section contents
# Tuple attribute names match those in mol2 spec (mol2.pdf)
#
from collections import namedtuple
MoleculeData = namedtuple("MoleculeData",
                            ["mol_name",      # string
                             "num_atoms",     # integer
                             "num_bonds",     # integer
                             "num_subst",     # integer
                             "num_feat",      # integer
                             "num_sets",      # integer
                             "mol_type",      # string
                             "charge_type",   # string
                             "status_bits",   # string
                             "mol_comment"])  # string
AtomData = namedtuple("AtomData",
                            ["atom_id",       # integer
                             "atom_name",     # string
                             "x", "y", "z",   # real
                             "atom_type",     # string
                             "subst_id",      # integer
                             "subst_name",    # string
                             "charge",        # real
                             "status_bit"])   # string
BondData = namedtuple("BondData",
                            ["bond_id",       # integer
                             "origin_atom_id",# integer
                             "target_atom_id",# integer
                             "bond_type",     # string
                             "status_bits"])  # string
SubstData = namedtuple("SubstData",
                            ["subst_id",      # integer
                             "subst_name",    # string
                             "root_atom",     # integer
                             "subst_type",    # string
                             "dict_type",     # integer
                             "chain",         # string
                             "sub_type",      # string
                             "inter_bonds",   # integer
                             "status",        # string
                             "comment"])      # string


# There is one MAJOR assumption in the parsing code:
# Comments ONLY occur before @<tripos>molecule sections
# and are associated with the following molecule.


class Mol2Parser:

    TriposPrefix = "@<tripos>"

    def __init__(self, session, stream, name, auto_style, atomic):
        self.session = session
        self.stream = stream
        self.name = name
        self.auto_style = auto_style
        self.atomic = atomic

        self.structures = []
        self._lineno = 0
        self._line = ""
        self._reset_structure()
        while self._read_section():
            pass
        self._check_gold()
        self._make_structure()

    def _read_section(self):
        """Read sections in mol2 file."""
        self._get_first_line()
        if self._line is None:
            return False
        import sys
        if self._is_section_tag():
            section_name = self._line[len(self.TriposPrefix):]
            if section_name.lower() == "molecule":
                if self._molecule is not None:
                    # Do not call if there is prelude data because
                    # the previous molecule, if any, has already been made
                    self._make_structure()
            self._get_line()    # Consume section line
            try:
                method = getattr(self, "_section_%s" % section_name.lower())
            except AttributeError:
                self._eat_section()
                self._warn("ignoring section '%s'" % section_name)
            else:
                method()
        elif self._line[0] == '#':
            self._section_prelude()
        else:
            self._warn("ignore unexpected line '%s'" % self._line)
            self._get_line()
        return True

    def _get_first_line(self):
        """Get first line in section, skipping blank lines."""
        if self._line is None:
            return None
        # Get first non-blank line
        while self._line == "":
            self._get_line()

    def _get_line(self):
        """Read the next line, stripping leading/trailing white space
        
        Current line is available in "_line" attribute."""
        line = self.stream.readline()
        if not line:
            self._line = None
        else:
            self._line = line.strip()
            self._lineno += 1

    def _is_section_tag(self):
        """Return if current line is a section tag"""
        return self._line.lower().startswith(self.TriposPrefix)

    def _warn(self, msg):
        """Print warning with current line number"""
        self.session.logger.warning("line %d: %s" % (self._lineno, msg))

    def _reset_structure(self):
        """Reset structure data cache"""
        self._data = {}
        self._molecule = None
        self._atoms = []
        self._bonds = []
        self._substs = []
        self._comments = []

    def _make_structure(self):
        """Build ChimeraX structure and reset structure data cache"""
        try:
            if self._molecule is None:
                return
            if self.atomic:
                from chimerax.atomic import AtomicStructure as SC
            else:
                from chimerax.atomic import Structure as SC
            # Create structure
            s = SC(self.session, auto_style=self.auto_style)
            s.name = self._molecule.mol_name
            SC.register_attr(self.session, "viewdockx_data", "ViewDockX")
            from chimerax.atomic import Atom
            Atom.register_attr(self.session, "charge", "ViewDockX", attr_type=float)
            Atom.register_attr(self.session, "mol2_type", "ViewDockX", attr_type=str)
            s.viewdockx_data = self._data
            if self._molecule.charge_type:
                s.charge_model = self._molecule.charge_type
            if self._molecule.mol_type:
                s.mol2_type = self._molecule.mol_type
            if self._molecule.mol_comment:
                s.mol2_comment = self._molecule.mol_comment
            # Create residues
            substid2residue = {}
            for subst_data in self._substs:
                # ChimeraX limitation: 4-letter residue type
                name = subst_data.subst_name[:4]
                chain = subst_data.chain
                if chain is None or chain == "****":
                    chain = ''
                residue = s.new_residue(name, chain, subst_data.subst_id)
                substid2residue[subst_data.subst_id] = residue
            # Create atoms
            atomid2atom = {}
            from numpy import array, float64
            for atom_data in self._atoms:
                name = atom_data.atom_name
                element = atom_data.atom_type
                if '.' in element:
                    element = element.split('.')[0]
                atom = s.new_atom(name, element)
                atom.coord = array([atom_data.x, atom_data.y, atom_data.z],
                                   dtype=float64)
                if atom_data.charge is not None:
                    atom.charge = atom_data.charge
                atom.mol2_type = atom_data.atom_type
                subst_id = atom_data.subst_id
                try:
                    residue = substid2residue[subst_id]
                except KeyError:
                    # Must not have been a substructure section
                    residue = s.new_residue("UNL", '', 1 if subst_id is None else subst_id)
                    substid2residue[atom_data.subst_id] = residue
                residue.add_atom(atom)
                atomid2atom[atom_data.atom_id] = atom
            # Create bonds
            for bond_data in self._bonds:
                try:
                    origin = atomid2atom[bond_data.origin_atom_id]
                    target = atomid2atom[bond_data.target_atom_id]
                except KeyError:
                    self.session.logger.warning("bad atom index in bond")
                else:
                    s.new_bond(origin, target)
            self.structures.append(s)
        finally:
            self._reset_structure()

    def _eat_section(self):
        """Consume all lines for current section"""
        # Stop on blank line/EOF, comment or @<tripos>
        self._get_line()
        while self._line:
            if self._line[0] == '#' or self._is_section_tag():
                break
            self._get_line()

    def _optional(self, parts, n, converter=None):
        """Return value for n'th argument if it exists"""
        try:
            v = parts[n]
        except IndexError:
            return None
        if converter:
            return converter(v)
        else:
            return v

    #
    # Section parsers.  Format specification in mol2.pdf.
    #
    def _section_prelude(self):
        # hash comment block before @<tripos>molecule
        self._make_structure()
        while self._line:
            if self._line[0] != '#':
                break
            non_hash = 0
            try:
                while self._line[non_hash] == '#':
                    non_hash += 1
            except IndexError:
                # line must be all #
                self._get_line()
                continue
            # Dock comments usually start with 10 #s
            # Dock 3.7 has lines with a single #, but they contain
            # a table of values that we cannot easily display, so skip
            if non_hash > 8:
                parts = self._line[non_hash:].lstrip().split(':', 1)
                if len(parts) != 2:
                    # Assume value is last field
                    parts = self._line[non_hash:].rsplit(None, 1)
                try:
                    self._data[parts[0].strip()] = parts[1].strip()
                except IndexError:
                    # Must be a single word on the line, just ignore
                    pass
            self._get_line()

    def _section_molecule(self):
        try:
            mol_name = self._line
            self._get_line()
            parts = self._line.split()
            if len(parts) < 2:
                raise ValueError("wrong number of fields in molecule data")
            num_atoms = int(parts[0])
            num_bonds = int(parts[1])
            num_subst = int(parts[2]) if len(parts) > 2 else 0
            num_feat = int(parts[3]) if len(parts) > 3 else 0
            num_sets = int(parts[4]) if len(parts) > 4 else 0
            self._get_line()
            mol_type = self._line
            self._get_line()
            charge_type = self._line
            self._get_line()
            if not self._is_section_tag():
                status_bits = self._line
                self._get_line()
            else:
                status_bits = None
            if not self._is_section_tag():
                mol_comment = self._line
                self._get_line()
            else:
                mol_comment = None
            self._molecule = MoleculeData(mol_name, num_atoms, num_bonds,
                                          num_subst, num_feat, num_sets,
                                          mol_type, charge_type, status_bits,
                                          mol_comment)
        except ValueError:
            # Must be integer conversion
            self._warn("bad molecule data")

    def _section_atom(self):
        # @<tripos>atom
        if self._molecule is None:
            self._eat_section()
            return
        try:
            for n in range(self._molecule.num_atoms):
                parts = self._line.split()
                atom_id = int(parts[0])
                atom_name = parts[1]
                x = float(parts[2])
                y = float(parts[3])
                z = float(parts[4])
                atom_type = parts[5]
                subst_id = self._optional(parts, 6, int)
                subst_name = self._optional(parts, 7)
                charge = self._optional(parts, 8, float)
                status_bit = self._optional(parts, 9)
                atom_data = AtomData(atom_id, atom_name, x, y, z, atom_type,
                                     subst_id, subst_name, charge, status_bit)
                self._atoms.append(atom_data)
                self._get_line()
        except ValueError:
            # Must be numeric conversion
            self._warn("bad atom data")

    def _section_bond(self):
        if self._molecule is None:
            self._eat_section()
            return
        try:
            for n in range(self._molecule.num_bonds):
                parts = self._line.split()
                bond_id = int(parts[0])
                origin_atom_id = int(parts[1])
                target_atom_id = int(parts[2])
                bond_type = parts[3]
                status_bits = self._optional(parts, 4)
                bond_data = BondData(bond_id, origin_atom_id, target_atom_id,
                                     bond_type, status_bits)
                self._bonds.append(bond_data)
                self._get_line()
        except ValueError:
            # Must be numeric conversion
            self._warn("bad bond data")

    def _section_substructure(self):
        if self._molecule is None:
            self._eat_section()
            return
        try:
            for n in range(self._molecule.num_subst):
                parts = self._line.split(None, 9)
                subst_id = int(parts[0])
                subst_name = parts[1]
                root_atom = int(parts[2])
                subst_type = self._optional(parts, 3)
                dict_type = self._optional(parts, 4, int)
                chain = self._optional(parts, 5)
                sub_type = self._optional(parts, 6)
                inter_bonds = self._optional(parts, 7, int)
                status = self._optional(parts, 8)
                comment = self._optional(parts, 9)
                subst_data = SubstData(subst_id, subst_name, root_atom,
                                       subst_type, dict_type, chain, sub_type,
                                       inter_bonds, status, comment)
                self._substs.append(subst_data)
                self._get_line()
        except ValueError:
            # Must be numeric conversion
            self._warn("bad substructure data")

    def _section_comment(self):
        if self._molecule is None:
            self._eat_section()
            return
        while self._line is not None:
            if self._is_section_tag():
                break
            self._comments.append(self._line)
            self._get_line()

    def _check_gold(self):
        import re
        re_gold = re.compile(r"> <Gold\.(?P<param>[^>]+)>\s*")
        fields = {}
        lines = None
        for line in self._comments:
            m = re_gold.match(line)
            if m is None:
                if lines is not None and line.strip():
                    lines.append(line)
            else:
                param = m.group("param")
                lines = []
                fields[param] = lines
        if fields:
            self._data = {"Name": self._molecule.mol_name.split("|")[0]}
            for param, lines in fields.items():
                if len(lines) == 1:
                    self._data[param] = _value(lines[0])


def _value(s):
    try:
        return int(s)
        return float(s)
    except ValueError:
        return s

def open_zdock(session, stream, file_name, auto_style, atomic):
    from chimerax.atomic import next_chain_id
    import tempfile
    out_f = tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', suffix=".pdb", delete=False)
    used_chains = set()
    cur_in_chain = cur_out_chain = cur_res_num = None
    for line in stream:
        line = line[:55]
        if line.startswith("ATOM  ") or line.startswith("HETATM"):
            chain_id = line[21]
            res_num = int(line[22:26].strip())
            if cur_res_num is None or cur_in_chain != chain_id or res_num < cur_res_num:
                # new chain, we think
                cur_out_chain = chain_id
                while cur_out_chain in used_chains:
                    cur_out_chain = next_chain_id(cur_out_chain)
                    if len(cur_out_chain) > 1:
                        raise IOError("Ran out of unique chain IDs")
                used_chains.add(cur_out_chain)
                cur_in_chain = chain_id
            cur_res_num = res_num
            if cur_out_chain != cur_in_chain:
                line = line[:21] + cur_out_chain + line[22:]
        print(line, file=out_f)
    out_f.close()
    from chimerax.pdb import open_pdb
    try:
        return open_pdb(session, out_f.name, file_name=file_name, auto_style=auto_style, atomic=atomic)
    finally:
        import os
        os.unlink(out_f.name)
