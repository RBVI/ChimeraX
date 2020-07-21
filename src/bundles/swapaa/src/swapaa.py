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


def swapaa(session, residues, restype):

    ures = residues.unique()
    if len(ures) == 0:
        from chimerax.core.errors import UserError
        raise UserError('swapaa: No residues specified')

    notaa = [r for r in ures if r.polymer_type != r.PT_PROTEIN]
    if notaa:
        from chimerax.core.errors import UserError
        raise UserError('swapaa: Cannot swap non-protein residues "%s"'
                        % ', '.join(r.string() for r in notaa))

    tres = template_residues(session)
    new_r = tres.residue(restype)
    for r in ures:
        swap_residue(r, new_r)
    
def template_residues(session):
    if not hasattr(session, '_swapaa_template_residues'):
        session._swapaa_template_residues = TemplateResidues(session)
    return session._swapaa_template_residues

class TemplateResidues:
    def __init__(self, session):
        self._residues = []			# List of Residue
        self._residue_names = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS',
                               'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
                               'LEU', 'LYS', 'MET', 'PHE', 'PRO',
                               'SER', 'THR', 'TRP', 'TYR', 'VAL']
        self._name_to_residue = {}		# Map 3 letter code to Residue
        self._template_file = 'templates.cif'	# Atom coordinates for each residue.
        self._load_templates(session)

    def list(self):
        return self._residues

    def residue(self, name):
        return self._name_to_residue.get(name)
    
    def _load_templates(self, session):
        tres = self._residues
        if tres:
            return
        from chimerax.mmcif import open_mmcif
        from os.path import join, dirname
        path = join(dirname(__file__), self._template_file)
        models, status = open_mmcif(session, path, log_info = False)
        
        m = models[0]
        found = self._name_to_residue
        tnames = self._residue_names
        for r in m.residues:
            if r.name not in found and r.name in tnames:
                found[r.name] = r
        tres.extend(found[name] for name in tnames if name in found)

def swap_residue(r, new_r,
                 align_atom_names = ['N', 'C', 'CA'],
                 keep_atom_names = ['N', 'C', 'CA', 'O', 'H']):
                 
    '''
    Replace residue r with a copy of template residue new_r, choosing the
    new residue position by aligning backbone atoms.
    '''
    pos, amap = _backbone_alignment(r, new_r, align_atom_names = align_atom_names)
    if pos is None:
        return False	# Missing backbone atoms to align new residue

    add_hydrogens = _has_hydrogens(r)
    carbon_color = _carbon_color(r)

    # Delete atoms.  Backbone atom HA is deleted if new residues is GLY.
    akeep = set(keep_atom_names).intersection(new_r.atoms.names)
    from chimerax.atomic import Atoms
    adel = Atoms([a for a in r.atoms if a.name not in akeep])
    adel.delete()

    # Create new atoms
    s = r.structure
    akept = set(r.atoms.names)
    # Set new atom b-factors to average of previous residue backbone atom b-factors.
    bbf = [a.bfactor for a in r.atoms]
    bfactor = sum(bbf)/len(bbf) if bbf else 0
    from chimerax.atomic.colors import element_color
    for a in new_r.atoms:
        if a.name not in akept:
            if a.element.name != 'H' or add_hydrogens:
                na = s.new_atom(a.name, a.element)
                na.scene_coord = pos * a.scene_coord
                # TODO: Color by element, but use model carbon color.
                na.color = carbon_color if a.element.name == 'C' else element_color(a.element.number)
                na.draw_mode = na.STICK_STYLE
                na.bfactor = bfactor
                r.add_atom(na)
                amap[a] = na
    
    # Create new bonds
    for b in new_r.atoms.intra_bonds:
        a1,a2 = b.atoms
        if a1 in amap and a2 in amap:
            na1, na2 = amap[a1], amap[a2]
            if not na1.connects_to(na2):
                nb = s.new_bond(na1, na2)

    # Set new residue name.
    r.name = new_r.name

    return True
        
def _backbone_alignment(r, new_r, align_atom_names = ['N', 'C', 'CA']):
    ra = dict((a.name, a) for a in r.atoms)
    nra = dict((a.name, a) for a in new_r.atoms)
    apairs = []
    for aname in align_atom_names:
        if aname in ra and aname in nra:
            apairs.append((ra[aname], nra[aname]))
    if len(apairs) < 3:
        log = r.structure.session.logger
        log.status('Fewer than 3 backbone atoms (%s) in residue %s (%s), swapaa cannot align'
                   % (', '.join(align_atom_names), str(r), ', '.join(ra.keys())), log = True)
        return None, None
    from chimerax.geometry import align_points
    from numpy import array
    xyz = array([a2.scene_coord for a1,a2 in apairs])
    ref_xyz = array([a1.scene_coord for a1,a2 in apairs])
    p, rms = align_points(xyz, ref_xyz)
    amap = dict((a2,a1) for a1,a2 in apairs)
    return p, amap
    
def _has_hydrogens(r):
    for a in r.atoms:
        if a.element.name == 'H':
            return True
    return False

def _carbon_color(r):
    for a in r.atoms:
        if a.element.name == 'C':
            return a.color
    from chimerax.atomic.colors import element_color
    return element_color(6)
         
def register_swapaa_command(logger):
    from chimerax.core.commands import CmdDesc, register, EnumOf
    from chimerax.atomic import ResiduesArg
    aa = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
          'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL']
    AminoAcidArg = EnumOf(aa)
    desc = CmdDesc(
        required = [('residues', ResiduesArg),
                    ('restype', AminoAcidArg),],
        synopsis = 'Replace residue with specified amino acid'
    )
    register('swapaa mousemode', desc, swapaa, logger=logger)
