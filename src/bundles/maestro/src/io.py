# vim: set expandtab shiftwidth=4 softtabstop=4:

def open_mae(session, path, file_name, auto_style, atomic):
    import os.path
    extension = os.path.splitext(path)[1]
    if extension == ".mae":
        with open(path, encoding="utf-8") as stream:
            p = MaestroParser(session, stream, file_name, auto_style, atomic)
    elif extension == ".maegz":
        import gzip
        with gzip.open(path, "rt") as stream:
            p = MaestroParser(session, stream, file_name, auto_style, atomic)
    structures = p.structures
    def num_atoms(s):
        try:
            return s.num_atoms
        except AttributeError:
            return sum([num_atoms(c) for c in s.child_models()])
    def num_bonds(s):
        try:
            return s.num_bonds
        except AttributeError:
            return sum([num_bonds(c) for c in s.child_models()])
    status = "Opened %s containing %d structures (%d atoms, %d bonds)" % (
                    file_name, len(structures),
                    sum([num_atoms(s) for s in structures]),
                    sum([num_bonds(s) for s in structures]))
    return structures, status


class MaestroParser:

    def __init__(self, session, stream, name, auto_style, atomic):
        from .maestro import MaestroFile
        from chimerax.core.errors import UserError
        self.session = session
        self.auto_style = auto_style
        self.atomic = atomic
        try:
            mf = MaestroFile(stream)
        except (ValueError, SyntaxError) as e:
            raise UserError(e.str())

        # Make sure we have the right type and version of data
        # from initial block
        mf_iter = iter(mf)
        block0 = next(mf_iter)
        try:
            if block0.get_attribute("s_m_m2io_version") != "2.0.0":
                raise ValueError("Maestro version mismatch")
            #print "Maestro v2.0.0 file recognized"
        except Exception:
            raise UserError("%s: not a v2.0.0 Maestro file" % path)

        # Convert all subsequent blocks named "f_m_ct" to molecules
        receptors = []
        ligands = []
        for block in mf_iter:
            if block.name != "f_m_ct":
                print("%s: Skipping \"%s\" block" % (name, block.name))
            #print "Convert %s block to molecule" % block.name
            s = self._make_structure(block)
            if s:
                try:
                    is_ligand = block.get_attribute("r_i_docking_score")
                except (KeyError, ValueError):
                    is_ligand = False
                if is_ligand:
                    ligands.append(s)
                else:
                    receptors.append(s)
                self._add_properties(s, block, is_ligand)
                s.name = name
        if not receptors:
            self.structures = ligands
        elif not ligands:
            self.structures = receptors
        else:
            from chimerax.core.models import Model
            self.structures = receptors
            container = Model(name, self.session)
            container.add(ligands)
            self.structures.append(container)

    def _make_structure(self, block):
        from numpy import array
        from .maestro import IndexAttribute
        if self.atomic:
            from chimerax.atomic import AtomicStructure as SC
        else:
            from chimerax.atomic import Structure as SC
        from chimerax.atomic import Element
        atoms = block.get_sub_block("m_atom")
        if atoms is None:
            print("No m_atom block found")
            return None
        bonds = block.get_sub_block("m_bond")
        s = SC(self.session, auto_style=self.auto_style)
        SC.register_attr(self.session, "viewdockx_data", "ViewDockX")

        residue_map = {}
        atom_map = {}
        for row in range(atoms.size):
            attrs = atoms.get_attribute_map(row)
            index = attrs[IndexAttribute]

            # Get residue data and create if necessary
            res_seq = attrs["i_m_residue_number"]
            insert_code = attrs.get("s_m_insertion_code", None)
            if not insert_code:
                insert_code = ' '
            chain_id = attrs.get("s_m_chain_name", ' ')
            res_key = (chain_id, res_seq, insert_code)
            try:
                r = residue_map[res_key]
            except KeyError:
                res_name = attrs.get("s_m_pdb_residue_name", "UNK").strip()
                r = s.new_residue(res_name, chain_id, res_seq, insert_code)
                residue_map[res_key] = r
            rgb = attrs.get("s_m_ribbon_color_rgb", None)
            if rgb:
                r.ribbon_color = self._get_color(rgb)

            # Get atom data and create
            try:
                name = attrs["s_m_pdb_atom_name"].strip()
            except KeyError:
                name = attrs.get("s_m_atom_name", "").strip()
            name = name.strip()
            atomic_number = attrs.get("i_m_atomic_number", 6)
            if atomic_number < 1:
                # Negative element number in bug #5087 that causes delayed crash.
                atomic_number = 1
            element = Element.get_element(atomic_number)
            if not name:
                name = element.name
            a = s.new_atom(name, element)
            a.coord = array([atoms.get_attribute("r_m_x_coord", row),
                             atoms.get_attribute("r_m_y_coord", row),
                             atoms.get_attribute("r_m_z_coord", row)])
            try:
                a.bfactor = attrs["r_m_pdb_tfactor"]
            except (KeyError, TypeError):
                a.bfactor = 0.0
            try:
                a.occupancy = attrs["r_m_pdb_occupancy"]
            except (KeyError, TypeError):
                a.occupancy = 1.0
            rgb = attrs.get("s_m_color_rgb", None)
            if rgb:
                a.color = self._get_color(rgb)

            # Add atom to residue and to atom map for bonding later
            r.add_atom(a)
            atom_map[index] = a
        if bonds is None or bonds.size == 0:
            s.connect_structure()
        else:
            for row in range(bonds.size):
                attrs = bonds.get_attribute_map(row)
                fi = attrs["i_m_from"]
                ti = attrs["i_m_to"]
                if ti < fi:
                    # Bonds are reported in both directions. We only need one.
                    continue
                afi = atom_map[fi]
                ati = atom_map[ti]
                b = s.new_bond(afi, ati)
                b.order = attrs["i_m_order"]
        return s

    def _add_properties(self, s, block, is_ligand):
        """Add properties to molecule."""
        from .maestro import get_value
        attrs = block.get_attribute_map()
        raw_text = []
        d = {}
        keep = set(["i", "m", "sd", "psp"])
        for key, value in attrs.items():
            is_valid = True
            # parts = key.split('_', 2)
            parts = self._split_key(key, 2)
            if len(parts) != 3:
                name = key
            else:
                if parts[1] not in keep:
                    is_valid = False
                name = parts[2]
            if is_valid:
                try:
                    converted_value = get_value(key, value)
                except ValueError:
                    is_valid = False
            if is_valid:
                d[name] = converted_value
            raw_text.append("%s: %s" % (name, value))
        s.maestro_text = '\n'.join(raw_text)
        if is_ligand:
            s.viewdockx_data = d

    def _split_key(self, key, limit):
        parts = []
        current = []
        escape = False
        for c in key:
            if escape:
                current.append(c)
                escape = False
            elif c == '\\':
                escape = True
            elif c == '_':
                if len(parts) < limit:
                    parts.append(''.join(current))
                    current = []
                else:
                    current.append(' ')
            else:
                current.append(c)
        if current:
            parts.append(''.join(current))
        return parts

    @staticmethod
    def _get_color(rgb):
        from numpy import array
        r = int(rgb[0:2], 16)
        g = int(rgb[2:4], 16)
        b = int(rgb[4:6], 16)
        return array([r, g, b, 255])
