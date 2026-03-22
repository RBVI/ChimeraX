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

# -----------------------------------------------------------------------------
#
from chimerax.core.commands import Annotation, AnnotationError, AtomSpecArg, ObjectsArg, ModelArg, StringArg

class AtomsArg(AtomSpecArg):
    """Parse command atoms specifier"""
    name = "an atoms specifier"

    @classmethod
    def parse(cls, text, session, ordered=False):
        aspec, text, rest = super().parse(text, session)
        atoms = aspec.evaluate(session, order_implicit_atoms=ordered).atoms
        atoms.spec = str(aspec)
        return atoms, text, rest


class AtomArg(AtomsArg):
    """Parse command specifier for an atom"""
    name = 'an atom specifier'

    @classmethod
    def parse(cls, text, session):
        atoms, used, rest = super().parse(text, session)
        if len(atoms) != 1:
            raise AnnotationError("must specify exactly one atom (specified %d)" % len(atoms))
        return atoms[0], used, rest


class ElementArg(StringArg):
    """Parse command specifier for an atomic symbol"""
    name = 'an atomic symbol'

    @classmethod
    def parse(cls, text, session):
        element_name, used, rest = super().parse(text, session)
        from . import Element
        e = Element.get_element(element_name)
        if e.number == 0:
            raise AnnotationError("'%s' is not an atomic symbol" % element_name)
        return e, used, rest


class OrderedAtomsArg(AtomsArg):

    @classmethod
    def parse(cls, text, session):
        return super().parse(text, session, ordered=True)


class ResiduesArg(AtomSpecArg):
    """Parse command residues specifier"""
    name = "a residues specifier"

    @classmethod
    def parse(cls, text, session):
        orig_text = text
        aspec, text, rest = super().parse(text, session)
        evaled = aspec.evaluate(session)
        outermost_inversion = aspec.outermost_inversion
        from .molarray import concatenate, Atoms, Residues
        # inter-residue bonds don't select either residue
        atoms1, atoms2 = evaled.bonds.atoms
        bond_atoms = atoms1.filter(atoms1.residues.pointers == atoms2.residues.pointers)
        atoms = concatenate((evaled.atoms, bond_atoms), Atoms)
        residues = atoms.residues.unique()
        if outermost_inversion:
            # the outermost operator was '~', so weed out partially-selected residues,
            # but generically weeding them out is very slow, so try a shortcut if possible...
            spec_text = orig_text[:len(orig_text) - len(rest)]
            if spec_text.count('~') == 1 or spec_text.strip().startswith('~'):
                uninverted_spec = spec_text.replace('~', '', 1)
                if cls.use_peglib_parser:
                    ui_evaled, *args = super().evaluate(session, uninverted_spec)
                else:
                    ui_aspec, *args = super().parse(uninverted_spec, session)
                    ui_evaled = ui_aspec.evaluate(session)
                ui_atoms1, ui_atoms2 = ui_evaled.bonds.atoms
                ui_bond_atoms = ui_atoms1.filter(ui_atoms1.residues.pointers == ui_atoms2.residues.pointers)
                ui_atoms = concatenate((ui_evaled.atoms, ui_bond_atoms), Atoms)
                ui_residues = ui_atoms.residues.unique()
                residues -= ui_residues
            else:
                if cls.use_peglib_parser:
                    explicit, *args = super().evaluate(session, text, add_implied=False)
                else:
                    explicit = aspec.evaluate(session, add_implied=False)
                unselected = residues.atoms - explicit.atoms
                residues = residues - unselected.residues.unique()
                # trickier to screen out partial bond selection, go residue by residue...
                remaining = []
                for r in residues:
                    res_bonds = r.atoms.intra_bonds
                    if len(res_bonds & explicit.bonds) == len(res_bonds):
                        remaining.append(r)
                residues = Residues(remaining)
        residues.spec = str(aspec)
        return residues, text, rest


class UniqueChainsArg(AtomSpecArg):
    """Parse command chains specifier"""
    name = "a chains specifier"

    @classmethod
    def parse(cls, text, session):
        aspec, text, rest = super().parse(text, session)
        chains = aspec.evaluate(session).atoms.residues.unique_chains
        outermost_inversion = aspec.outermost_inversion
        if outermost_inversion:
            # the outermost operator was '~', so weed out partially-selected residues
            if cls.use_peglib_parser:
                explicit, *args = super().evaluate(session, text, add_implied=False)
            else:
                explicit = aspec.evaluate(session, add_implied=False)
            remaining = []
            for chain in chains:
                chain_atoms = chain.existing_residues.atoms
                if len(chain_atoms) != len(chain_atoms & explicit.atoms):
                    continue
                chain_bonds = chain_atoms.intra_bonds
                if len(chain_bonds & explicit.bonds) == len(chain_bonds):
                    remaining.append(chain)
            from .molarray import Chains
            chains = Chains(remaining)
        chains.spec = str(aspec)
        return chains, text, rest


class ChainArg(UniqueChainsArg):
    """Parse command chains specifier"""
    name = "a chains specifier"

    @classmethod
    def parse(cls, text, session):
        chains, text, rest = super().parse(text, session)
        if len(chains) != 1:
            raise AnnotationError("must specify exactly one chain")
        return chains[0], text, rest


class SequencesArg(Annotation):
    '''
    Accept a chain atom spec (#1/A), a sequence viewer alignment id (myseqs.aln:2),
    a UniProt accession id (K9Z9J3, 6 or 10 characters, always has numbers),
    a UniProt name (MYOM1_HUMAN, always has underscore, X_Y where X and Y are at most
    5 alphanumeric characters), or a sequence (MVLSPADKTN....).
    Returns a list of Sequence objects or or objects derived from Sequence such as Chain.
    '''
    name = 'sequences'
    
    @classmethod
    def parse(cls, text, session):
        if is_atom_spec(text, session):
            return UniqueChainsArg.parse(text, session)

        if not text:
            raise AnnotationError("Expected %s" % cls.name)
        from chimerax.core.commands import next_token
        token, used, rest = next_token(text)
        if len(text) == 0:
            raise AnnotationError('Sequences argument is empty.')

        seqs = []
        for seq_text in token.split(','):
            seq = _parse_sequence(seq_text, session)
            if seq is None:
                raise AnnotationError('Sequences argument "%s" is not a chain specifier, ' % seq_text +
                                      'alignment id, UniProt id, or sequence characters')
            seqs.append(seq)

        return seqs, used, rest

def _parse_sequence(seq_text, session):
    if is_uniprot_id(seq_text, first_token_only = True):
        seq, sused, srest = UniProtSequenceArg.parse(seq_text, session)
        if len(srest) == 0:
            return seq
            
    from chimerax.seqalign import SeqArg
    for arg_type in (ChainArg, SeqArg, RawSequenceArg):
        try:
            seq, sused, srest = arg_type.parse(seq_text, session)
            if len(srest) == 0:
                return seq
        except Exception:
            pass
    return None

class SequenceArg(Annotation):
    name = 'sequence'
    
    @classmethod
    def parse(cls, text, session):
        value, used, rest = SequencesArg.parse(text, session)
        if len(value) != 1:
            raise AnnotationError('Sequences argument "%s" must specify 1 sequence, got %d'
                                  % (used, len(value)))
        return value[0], used, rest
    
def is_atom_spec(text, session):
    try:
        if AtomSpecArg.use_peglib_parser:
            AtomSpecArg.evaluate(session, text)
        else:
            AtomSpecArg.parse(text, session)
    except AnnotationError:
        return False
    return True
                
class UniProtSequenceArg(Annotation):
    name = 'UniProt sequence'
    
    @classmethod
    def parse(cls, text, session):
        uid, used, rest = StringArg.parse(text, session)
        if not is_uniprot_id(uid):
            raise AnnotationError('Invalid UniProt identifier "%s"' % uid)
        if '_' in uid:
            uname = uid
            from chimerax.uniprot import map_uniprot_ident
            try:
                uid = map_uniprot_ident(uid, return_value = 'entry')
            except Exception as e:
                raise AnnotationError(f'Could not look up UniProt accession code for "{uid}". {e}')
        else:
            uname = None
        if len(uid) not in (6, 10):
            raise AnnotationError('UniProt id "%s" must be 6 or 10 characters' % uid)
        from chimerax.uniprot.fetch_uniprot import fetch_uniprot_accession_info
        try:
            seq_string, full_name, features = fetch_uniprot_accession_info(session, uid)
        except Exception as e:
            raise AnnotationError(f'Could not fetch sequence for UniProt id "{uid}". {e}')
        from . import Sequence
        seq = Sequence(name = (uname or uid), characters = seq_string)
        seq.uniprot_accession = uid
        if uname is not None:
            seq.uniprot_name = uname
        return seq, used, rest
                
class UniProtIdArg(Annotation):
    name = 'UniProt id'
    
    @classmethod
    def parse(cls, text, session):
        uid, used, rest = StringArg.parse(text, session)
        if not is_uniprot_id(uid):
            raise AnnotationError('Invalid UniProt identifier "%s"' % uid)
        if '_' in uid:
            from chimerax.uniprot import map_uniprot_ident
            try:
                uid = map_uniprot_ident(uid, return_value = 'entry')
            except Exception:
                raise AnnotationError('UniProt name "%s" must be 1-5 characters followed by an underscore followed by 1-5 characters' % uid)
        if len(uid) not in (6, 10):
            raise AnnotationError('UniProt id "%s" must be 6 or 10 characters' % uid)
        return uid.upper(), used, rest

def is_uniprot_id(id, first_token_only = False):
    # Name and accession format described here.
    # https://www.uniprot.org/help/accession_numbers
    # https://www.uniprot.org/help/entry_name
    if len(id) == 0:
        return False
    if first_token_only:
        from chimerax.core.commands import next_token
        id, text, rest = next_token(id)
    if '_' in id:
        fields = id.split('_')
        f1,f2 = fields
        if (f1.isalnum() and (len(f1) <= 6 or len(f1) == 10) and
            f2.isalnum() and len(f2) <= 5):
            return True
    elif (len(id) >= 6 and id.isalnum() and
          id[0].isalpha() and id[1].isdigit() and id[5].isdigit()):
        if len(id) == 6:
            return True
        elif len(id) == 10 and id[6].isalpha() and id[9].isdigit():
            return True
    return False

class RawSequenceArg(Annotation):
    name = 'sequence string'
    
    @classmethod
    def parse(cls, text, session):
        seqchars, used, rest = StringArg.parse(text, session)
        upper_a_to_z = set('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
        if not set(seqchars).issubset(upper_a_to_z):
            nonalpha = ''.join(set(seqchars) - upper_a_to_z)
            raise AnnotationError('Sequence "%s" contains characters "%s" that are not upper case A to Z.'
                                  % (seqchars, nonalpha))
        from . import Sequence
        seq = Sequence(characters = seqchars)
        return seq, used, rest

    
def fully_selected(session, explicit, mols):
    return [m for m in mols
        if m.num_atoms == len(explicit.atoms & m.atoms) and m.num_bonds == len(explicit.bonds & m.bonds)]

class StructuresArg(AtomSpecArg):
    """Parse command structures specifier"""
    name = "a structures specifier"

    @classmethod
    def parse(cls, text, session):
        if cls.use_peglib_parser:
            objs, text, rest = super().evaluate(session, text)
            models = objs.models
            outermost_inversion = objs.outermost_inversion
        else:
            aspec, text, rest = super().parse(text, session)
            models = aspec.evaluate(session).models
            outermost_inversion = aspec.outermost_inversion
        from . import Structure
        mols = [m for m in models if isinstance(m, Structure)]
        if outermost_inversion:
            if cls.use_peglib_parser:
                explicit, *args = super().evaluate(session, text, add_implied=False)
            else:
                explicit = aspec.evaluate(session, add_implied=False)
            mols = fully_selected(session, explicit, mols)
        return mols, text, rest


class AtomicStructuresArg(AtomSpecArg):
    """Parse command atomic structures specifier"""
    name = "an atomic structures specifier"

    @classmethod
    def parse(cls, text, session):
        if cls.use_peglib_parser:
            objs, text, rest = super().evaluate(session, text)
            models = objs.models
            outermost_inversion = objs.outermost_inversion
        else:
            aspec, text, rest = super().parse(text, session)
            models = aspec.evaluate(session).models
            outermost_inversion = aspec.outermost_inversion
        from . import AtomicStructure, AtomicStructures
        mols = [m for m in models if isinstance(m, AtomicStructure)]
        if outermost_inversion:
            if cls.use_peglib_parser:
                explicit, *args = super().evaluate(session, text, add_implied=False)
            else:
                explicit = aspec.evaluate(session, add_implied=False)
            mols = fully_selected(session, explicit, mols)
        return AtomicStructures(mols), text, rest


class AtomicStructureArg(AtomSpecArg):
    """Parse command atomic structure specifier"""
    name = "an atomic structure specifier"

    @classmethod
    def parse(cls, text, session):
        if cls.use_peglib_parser:
            objs, text, rest = super().evaluate(session, text)
            models = objs.models
            outermost_inversion = objs.outermost_inversion
        else:
            aspec, text, rest = super().parse(text, session)
            models = aspec.evaluate(session).models
            outermost_inversion = aspec.outermost_inversion
        from . import AtomicStructure
        mols = [m for m in models if isinstance(m, AtomicStructure)]
        if outermost_inversion:
            if cls.use_peglib_parser:
                explicit, *args = super().evaluate(session, text, add_implied=False)
            else:
                explicit = aspec.evaluate(session, add_implied=False)
            mols = fully_selected(session, explicit, mols)
        if len(mols) != 1:
            raise AnnotationError('must specify 1 atomic structure, got %d for "%s"'
                                  % (len(mols), text))
        return mols[0], text, rest


class PseudobondGroupsArg(AtomSpecArg):
    """Parse command atom specifier for pseudobond groups"""
    name = 'a pseudobond groups specifier'

    @classmethod
    def parse(cls, text, session):
        if cls.use_peglib_parser:
            objs, used, rest = super().evaluate(session, text)
            models = objs.models
        else:
            value, used, rest = super().parse(text, session)
            models = value.evaluate(session).models
        from . import PseudobondGroup
        pbgs = [m for m in models if isinstance(m, PseudobondGroup)]
        return pbgs, used, rest


class PseudobondsArg(ObjectsArg):
    """Parse command specifier for pseudobonds"""
    name = 'a pseudobonds specifier'

    @classmethod
    def parse(cls, text, session):
        objects, used, rest = super().parse(text, session)
        from . import interatom_pseudobonds, Pseudobonds, concatenate
        apb = interatom_pseudobonds(objects.atoms)
        opb = objects.pseudobonds
        pbonds = concatenate([apb, opb], Pseudobonds, remove_duplicates=True)
        return pbonds, used, rest


class BondsArg(ObjectsArg):
    """Parse command specifier for bonds"""
    name = 'a bonds specifier'

    @classmethod
    def parse(cls, text, session):
        objects, used, rest = super().parse(text, session)
        bonds = objects.bonds
        return bonds, used, rest


class BondArg(BondsArg):
    """Parse command specifier for a bond"""
    name = 'a bond specifier'

    @classmethod
    def parse(cls, text, session):
        bonds, used, rest = super().parse(text, session)
        if len(bonds) != 1:
            raise AnnotationError("must specify exactly one bond (specified %d)" % len(bonds))
        return bonds[0], used, rest

class StructureArg(ModelArg):
    """Parse command structure specifier"""
    name = "a structure specifier"

    @classmethod
    def parse(cls, text, session):
        m, text, rest = super().parse(text, session)
        from . import Structure
        models = [s for s in m.all_models() if isinstance(s, Structure)]
        if len(models) != 1:
            raise AnnotationError('must specify 1 structure, got %d for "%s"' % (len(models), text))
        return models[0], text, rest


class SymmetryArg(Annotation):
    '''Symmetry specification, e.g. C3 or D7'''
    name = 'symmetry'

    @staticmethod
    def parse(text, session):
        from chimerax.core.commands import next_token
        group, atext, rest = next_token(text)
        return Symmetry(group, session), atext, rest

# -----------------------------------------------------------------------------
#
class Symmetry:
    def __init__(self, group, session):
        self.group = group
        self.session = session
    def positions(self, center = None, axis = None, coord_sys = None,
                  molecule = None):
        '''
        center is a Center object.
        axis is an Axis object.
        coord_sys is a Place object mapping center/axis coordinates to scene coordinates,
        molecule is for biomt symmetries.
        Returned positions is a Places object in scene coordinates.
        '''
        if center:
            c = center.scene_coordinates(coord_sys)
        elif axis and axis.base_point() is not None:
            c = axis.base_point()
        elif coord_sys:
            c = coord_sys.origin()
        else:
            c = None
        if axis:
            a = axis.scene_coordinates(coord_sys)
        elif coord_sys:
            a = coord_sys.z_axis()
        else:
            a = None
        return parse_symmetry(self.session, self.group, c, a, molecule)

# Molecule arg is used for biomt symmetry.
def parse_symmetry(session, group, center = None, axis = None, molecule = None):

    # Handle products of symmetry groups.
    groups = group.split('*')
    from chimerax.geometry import Places
    ops = Places()
    for g in groups:
        ops = ops * group_symmetries(session, g, molecule)

    # Apply center and axis transformation.
    if center is not None or axis is not None:
        from chimerax.geometry import Place, vector_rotation, translation
        tf = Place()
        if center is not None and tuple(center) != (0,0,0):
            tf = translation([-c for c in center])
        if axis is not None and tuple(axis) != (0,0,1):
            tf = vector_rotation(axis, (0,0,1)) * tf
        if not tf.is_identity():
            ops = ops.transform_coordinates(tf)
    return ops

# -----------------------------------------------------------------------------
#
def group_symmetries(session, group, molecule):

    from chimerax import geometry
    from chimerax.core.errors import UserError

    g0 = group[:1].lower()
    gfields = group.split(',')
    g = gfields[0].lower()
    nf = len(gfields)
    recenter = True
    if g0 in ('c', 'd'):
        # Cyclic or dihedral symmetry: C<n>, D<n>
        try:
            n = int(group[1:])
        except ValueError:
            raise UserError('Invalid symmetry group syntax "%s"' % group)
        if n < 1:
            raise UserError('Cn or Dn with n = %d < 1' % (n,))
        if g0 == 'c':
            tflist = geometry.cyclic_symmetry_matrices(n)
        else:
            tflist = geometry.dihedral_symmetry_matrices(n)
    elif g == 'i':
        # Icosahedral symmetry: i[,<orientation>]
        if nf == 1:
            orientation = '222'
        elif nf == 2:
            orientation = gfields[1]
            if not orientation in geometry.icosahedral_orientations:
                raise UserError('Unknown icosahedron orientation "%s"'
                                   % orientation)
        else:
            raise UserError('Invalid symmetry group syntax "%s"' % group)
        tflist = geometry.icosahedral_symmetry_matrices(orientation)
    elif g == 't' and nf <= 2:
        # Tetrahedral symmetry t[,<orientation]
        if nf == 1:
            orientation = '222'
        elif nf == 2:
            orientation = gfields[1]
            if not orientation in geometry.tetrahedral_orientations:
                tos = ', '.join(geometry.tetrahedral_orientations)
                raise UserError('Unknown tetrahedral symmetry orientation %s'
                                   ', must be one of %s' % (gfields[1], tos))
        else:
            raise UserError('Invalid symmetry group syntax "%s"' % group)
        tflist = geometry.tetrahedral_symmetry_matrices(orientation)
    elif g == 'o':
        # Octahedral symmetry
        if nf == 1:
            tflist = geometry.octahedral_symmetry_matrices()
        else:
            raise UserError('Invalid symmetry group syntax "%s"' % group)
    elif g == 'h':
        # Helical symmetry: h,<rise>,<angle>,<n>[,<offset>]
        if nf < 4 or nf > 5:
            raise UserError('Invalid symmetry group syntax "%s"' % group)
        try:
            param = [float(f) for f in gfields[1:]]
        except ValueError:
            raise UserError('Invalid symmetry group syntax "%s"' % group)
        if len(param) == 3:
            param.append(0.0)
        rise, angle, n, offset = param
        n = int(n)
        from chimerax.geometry import Places
        tflist = Places([geometry.helical_symmetry_matrix(rise, angle, n = i+offset)
                         for i in range(n)])
    elif gfields[0].lower() == 'shift' or (g0 == 't' and nf >= 3):
        # Translation symmetry: t,<n>,<distance> or t,<n>,<dx>,<dy>,<dz>
        if nf != 3 and nf != 5:
            raise UserError('Invalid symmetry group syntax "%s"' % group)
        try:
            param = [float(f) for f in gfields[1:]]
        except ValueError:
            raise UserError('Invalid symmetry group syntax "%s"' % group)
        n = param[0]
        if n != int(n):
            raise UserError('Invalid symmetry group syntax "%s"' % group)
        n = int(n)
        if nf == 3:
          delta = (0,0,param[1])
        else:
          delta = param[1:]
        tflist = geometry.translation_symmetry_matrices(n, delta)
    elif group.lower() == 'biomt':
        # Biological unit
        from . import biological_unit_matrices
        tflist = biological_unit_matrices(molecule)
        if len(tflist) == 0:
            raise UserError('Molecule %s has no biological unit info' % molecule.name)
        if len(tflist) == 1 and tflist[0].is_identity():
            log = session.logger
            log.status('Molecule %s is the biological unit' % molecule.name)
        tflist = tflist.transform_coordinates(molecule.position.inverse())
        recenter = False
    elif g0 == '#':
        from chimerax.core.commands import ModelsArg
        if nf == 1:
            models = ModelsArg.parse(group, session)[0]
            mslist = [model_symmetry(m) for m in models]
            mslist = [ms for ms in mslist if ms is not None]
            if len(mslist) == 0:
                raise UserError('No symmetry for "%s"' % group)
            elif len(mslist) > 1:
                raise UserError('Multiple models "%s"' % group)
            tflist = mslist[0]
            recenter = False
        elif nf == 2:
            gf0, gf1 = gfields
            models = ModelsArg.parse(gf0, session)[0]
            mlist = [m for m in models
                     if hasattr(m, 'placements') and callable(m.placements)]
            if len(mlist) == 0:
                raise UserError('No placements for "%s"' % gf0)
            elif len(mlist) > 1:
                raise UserError('Multiple models with placements "%s"' % gf0)
            m = mlist[0]
            tflist = m.placements(gf1)
            if len(tflist) == 0:
                raise UserError('No placements "%s" for "%s"' % (gf1, gf0))
            c = molecule.atoms.coords.mean(axis = 0)
            cg = molecule.position * c
            cm = m.position.inverse() * cg
            tflist = make_closest_placement_identity(tflist, cm)
            recenter = False
    else:
        raise UserError('Unknown symmetry group "%s"' % group)

    return tflist

# -----------------------------------------------------------------------------
#
def model_symmetry(model):

    from chimerax.map import Volume
    from . import Structure
    if isinstance(model, Volume):
        tflist = model.data.symmetries
    elif isinstance(model, Structure):
        from . import biological_unit_matrices
        tflist = biological_unit_matrices(model)
    else:
        tflist = []

    if len(tflist) <= 1:
        return None

    return tflist

# -----------------------------------------------------------------------------
# Find the transform that maps (0,0,0) closest to the molecule center and
# multiply all transforms by the inverse of that transform.  This chooses
# the best placement for the current molecule position and makes all other
# placements relative to that one.
#
def make_closest_placement_identity(tflist, center):

    d = tflist.array()[:,:,3] - center
    d2 = (d*d).sum(axis = 1)
    i = d2.argmin()
    tfinv = tflist[i].inverse()
    rtflist = [tf*tfinv for tf in tflist]
    from chimerax.geometry import Place, Places
    rtflist[i] = Place()
    return Places(rtflist)

def _form_range(items, index_map, str_func):
    range_string = ""
    start_range = end_range = items[0]
    for item in items[1:]:
        ii, ei = index_map[item], index_map[end_range]
        if ii == ei:
            continue
        if ii == ei + 1:
            end_range = item
        else:
            if range_string:
                range_string += ','
            if start_range == end_range:
                range_string += str_func(start_range)
            else:
                range_string += str_func(start_range) + '-' + str_func(end_range)
            start_range = end_range = item
    if range_string:
        range_string += ','
    if start_range == end_range:
        range_string += str_func(start_range)
    else:
        range_string += str_func(start_range) + '-' + str_func(end_range)
    return range_string

def concise_residue_spec(session, residues):
    from . import Residues
    if not isinstance(residues, Residues):
        residues = Residues(residues)
    from . import all_structures
    need_model_spec = len(all_structures(session)) > 1
    full_spec = ""
    for struct, struct_residues in residues.by_structure:
        res_index_map = {}
        for i, r in enumerate(struct.residues):
            res_index_map[r] = i
        chain_id_index_map = {}
        for i, cid in enumerate(sorted(struct.residues.unique_chain_ids)):
            chain_id_index_map[cid] = i
        specs = {}
        for struct, chain_id, chain_residues in struct_residues.by_chain:
            # if chain_residues is all the residues with that chain ID, don't need a residue spec
            if len(struct.residues[struct.residues.chain_ids == chain_id]) == len(chain_residues):
                spec = ""
            else:
                sort_residues = list(chain_residues)
                sort_residues.sort(key=lambda res: (res.number, res.insertion_code))
                spec = ':' + _form_range(sort_residues, res_index_map, lambda r:
                    r.string(omit_structure=True, style="command", residue_only=True)[1:])
            specs.setdefault(spec, []).append(chain_id)

        if full_spec:
            full_spec += ' '
        spec_chain_ids = list(specs.items())
        spec_chain_ids.sort(key=lambda spec_cids: sorted(spec_cids[1])[0])
        if need_model_spec:
            full_spec += struct.string(style="command")
        for spec, chain_ids in spec_chain_ids:
            if ' ' in chain_ids:
                full_spec += '/*' + spec
            else:
                full_spec += '/' + _form_range(chain_ids, chain_id_index_map, str) + spec

    return full_spec

def concise_chain_spec(chains):
    schains = {}
    for c in chains:
        s = c.structure
        if s not in schains:
            schains[s] = set()
        schains[s].add(c)
    specs = []
    for structure, cset in schains.items():
        cids = sorted([c.chain_id for c in cset])
        specs.append(structure.string(style = 'command') + '/' + ','.join(cids))
    spec = ''.join(specs)
    return spec


def classify_hydrogen_visibility(displayed_atoms, all_atoms):
    """Classify the hydrogen visibility pattern for a set of atoms.
    
    Compares displayed hydrogen atoms against available hydrogen atoms
    in residues that have displayed atoms, to determine which visibility 
    pattern applies.
    
    Parameters
    ----------
    displayed_atoms : Atoms
        The atoms that are currently displayed
    all_atoms : Atoms
        All atoms in the context (e.g., all atoms in a chain)
        
    Returns
    -------
    str
        One of:
        - 'none': No hydrogens are shown
        - 'polar': Only polar hydrogens (H & ~HC) are shown
        - 'all': All hydrogens are shown
        - 'some': An arbitrary subset of hydrogens is shown
    """
    from . import Atoms
    if not isinstance(displayed_atoms, Atoms):
        displayed_atoms = Atoms(displayed_atoms)
    if not isinstance(all_atoms, Atoms):
        all_atoms = Atoms(all_atoms)
    
    # Get displayed hydrogen atoms
    displayed_h = displayed_atoms.filter(displayed_atoms.elements.numbers == 1)
    num_displayed_h = len(displayed_h)
    
    if num_displayed_h == 0:
        return 'none'
    
    # Get residues that have displayed atoms - we only consider H from these
    displayed_residues = displayed_atoms.unique_residues
    
    # Get all hydrogen atoms from residues that have displayed atoms
    # This ensures we compare against the relevant scope, not the entire chain
    residue_atoms = displayed_residues.atoms
    relevant_h = residue_atoms.filter(residue_atoms.elements.numbers == 1)
    
    if len(relevant_h) == 0:
        # No hydrogens exist in displayed residues
        return 'none'
    
    num_relevant_h = len(relevant_h)
    
    if num_displayed_h == num_relevant_h:
        return 'all'
    
    # Check if only polar hydrogens are shown
    # Polar H have idatm_type != 'HC' (HC = hydrogen bonded to carbon)
    polar_h = relevant_h.filter(relevant_h.idatm_types != 'HC')
    num_polar_h = len(polar_h)
    
    if num_polar_h > 0 and num_displayed_h == num_polar_h:
        # Check that the displayed H are exactly the polar H from displayed residues
        polar_h_set = set(polar_h._pointers)
        displayed_h_set = set(displayed_h._pointers)
        if polar_h_set == displayed_h_set:
            return 'polar'
    
    # Otherwise, it's an arbitrary subset
    return 'some'


def concise_atom_spec(session, atoms, include_hydrogens=False):
    """Generate concise atomspec for atoms.
    
    Uses residue-level specs when all non-hydrogen atoms of a residue are shown,
    and atom-level specs (residue@atom1,atom2,...) when only some non-H atoms are shown.
    
    Example output: "#1/A:103@CA,CB,C,N,O:110,112"
    - Residue 103 has only CA, CB, C, N, O shown (partial)
    - Residues 110, 112 have all non-H atoms shown (full)
    
    Parameters
    ----------
    session : Session
        The ChimeraX session
    atoms : Atoms
        Collection of atoms to generate spec for
    include_hydrogens : bool
        If True, include hydrogen atoms in the spec. If False (default),
        hydrogens are excluded and only non-H atoms determine full/partial display.
        
    Returns
    -------
    str
        Concise atomspec string
    """
    from . import Atoms
    if not isinstance(atoms, Atoms):
        atoms = Atoms(atoms)
    
    if len(atoms) == 0:
        return ""
    
    from . import all_structures
    need_model_spec = len(all_structures(session)) > 1
    
    # Create a set of atom pointers for fast lookup
    atom_set = set(atoms._pointers)
    
    full_spec = ""
    
    for struct, struct_atoms in atoms.by_structure:
        # Build index maps for range formation
        res_index_map = {}
        for i, r in enumerate(struct.residues):
            res_index_map[r] = i
        chain_id_index_map = {}
        for i, cid in enumerate(sorted(struct.residues.unique_chain_ids)):
            chain_id_index_map[cid] = i
        
        # Group atoms by chain
        chain_data = {}  # chain_id -> list of (residue, atom_names or None)
        
        for struct, chain_id, chain_atoms in struct_atoms.by_chain:
            residue_info = []  # list of (residue, atom_names or None)
            
            for res in chain_atoms.unique_residues:
                res_atoms = res.atoms
                
                # Get which of our input atoms are in this residue
                res_input_atoms = [a for a in res_atoms if a._c_pointer.value in atom_set]
                
                if include_hydrogens:
                    # Include all atoms (including H) in determination
                    all_res_atoms = res_atoms
                    res_input_relevant = res_input_atoms
                else:
                    # Only consider non-hydrogen atoms
                    all_res_atoms = res_atoms.filter(res_atoms.elements.numbers > 1)
                    res_input_relevant = [a for a in res_input_atoms if a.element.number > 1]
                
                if len(res_input_relevant) == 0:
                    # No relevant atoms from this residue
                    continue
                
                # Check if all relevant atoms are in our input
                if len(res_input_relevant) >= len(all_res_atoms):
                    # Full display - just residue spec needed
                    residue_info.append((res, None))
                else:
                    # Partial display - need atom names
                    if include_hydrogens:
                        atom_names = sorted(set(a.name for a in res_input_atoms))
                    else:
                        atom_names = sorted(set(a.name for a in res_input_atoms if a.element.number > 1))
                    residue_info.append((res, atom_names))
            
            if residue_info:
                # Sort by residue number and insertion code
                residue_info.sort(key=lambda ri: (ri[0].number, ri[0].insertion_code))
                chain_data[chain_id] = residue_info
        
        if not chain_data:
            continue
        
        # Build spec for this structure
        if full_spec:
            full_spec += ' '
        if need_model_spec:
            full_spec += struct.string(style="command")
        
        # Process each chain
        sorted_chain_ids = sorted(chain_data.keys(), key=lambda cid: chain_id_index_map.get(cid, 0))
        for chain_id in sorted_chain_ids:
            residue_info = chain_data[chain_id]
            
            # Check if all residues in this chain are full display
            all_full = all(atom_names is None for res, atom_names in residue_info)
            
            if all_full:
                # Use concise residue spec format (ranges)
                residues = [res for res, _ in residue_info]
                all_chain_res = struct.residues.filter(struct.residues.chain_ids == chain_id)
                if len(residues) == len(all_chain_res):
                    # All residues in chain - just chain spec
                    full_spec += '/' + chain_id
                else:
                    res_spec = _form_range(residues, res_index_map, lambda r:
                        r.string(omit_structure=True, style="command", residue_only=True)[1:])
                    full_spec += '/' + chain_id + ':' + res_spec
            else:
                # Mix of full and partial - need to handle each segment
                full_spec += '/' + chain_id
                full_spec += _build_mixed_residue_spec(residue_info, res_index_map)
    
    return full_spec


def _build_mixed_residue_spec(residue_info, res_index_map):
    """Build residue spec when there's a mix of full and partial display residues.
    
    Parameters
    ----------
    residue_info : list of (Residue, atom_names or None)
        List of residues with their atom names (None means full display)
    res_index_map : dict
        Map from residue to index for range formation
        
    Returns
    -------
    str
        Residue spec portion (starting with ':')
    """
    if not residue_info:
        return ""
    
    spec = ""
    i = 0
    while i < len(residue_info):
        res, atom_names = residue_info[i]
        
        if atom_names is not None:
            # Partial display - output with atom names
            res_str = res.string(omit_structure=True, style="command", residue_only=True)
            atom_str = ','.join(atom_names)
            spec += res_str + '@' + atom_str
            i += 1
        else:
            # Full display - collect consecutive full residues for range
            full_start = i
            while i < len(residue_info) and residue_info[i][1] is None:
                i += 1
            full_residues = [r for r, _ in residue_info[full_start:i]]
            
            # Use _form_range for consecutive full residues
            res_spec = _form_range(full_residues, res_index_map, lambda r:
                r.string(omit_structure=True, style="command", residue_only=True)[1:])
            spec += ':' + res_spec
    
    return spec
