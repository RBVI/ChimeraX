from chimera2 import UserError, scene
from chimera2.math3d import Point, Xform, Identity, weighted_point

_builtin_open = open

def open(stream, *args, **kw):

	if hasattr(stream, 'read'):
		input = stream
	else:
		# it's really a filename
		input = _builtin_open(stream, 'rb')

	import pdbio, access
	mol_blob = pdbio.read_pdb_file(input)
	atom_blob, bond_list = access.atoms_bonds(mol_blob)
	coords = access.coords(atom_blob)
	element_numbers = access.element_numbers(atom_blob)

	for center, en in zip(coords, element_numbers):
		radius = element_radius(en)
		rgb = element_color(en)
		color = [x / 255.0 for x in rgb] + [1]
		scene.add_sphere(radius, Point(center), color, None)

	for a0, a1 in bond_list:
		radius = 0.2
		p0 = Point(coords[a0])
		rgb0 = element_color(element_numbers[a0])
		color0 = [x / 255.0 for x in rgb0] + [1]
		p1 = Point(coords[a1])
		rgb1 = element_color(element_numbers[a1])
		color1 = [x / 255.0 for x in rgb1] + [1]
		mid = weighted_point([p0, p1])
		scene.add_cylinder(radius, p0, mid, color0, None)
		scene.add_cylinder(radius, p1, mid, color1, None)

	if input != stream:
		input.close()

_element_colors = dict(enumerate([
	[255, 255, 255], [217, 255, 255], [204, 128, 255], [194, 255, 0],
	[255, 181, 181], [144, 144, 144], [48, 80, 248], [255, 13, 13],
	[144, 224, 80], [179, 227, 245], [171, 92, 242], [138, 255, 0],
	[191, 166, 166], [240, 200, 160], [255, 128, 0], [255, 255, 48],
	[31, 240, 31], [128, 209, 227], [143, 64, 212], [61, 255, 0],
	[230, 230, 230], [191, 194, 199], [166, 166, 171], [138, 153, 199],
	[156, 122, 199], [224, 102, 51], [240, 144, 160], [80, 208, 80],
	[200, 128, 51], [125, 128, 176], [194, 143, 143], [102, 143, 143],
	[189, 128, 227], [255, 161, 0], [166, 41, 41], [92, 184, 209],
	[112, 46, 176], [0, 255, 0], [148, 255, 255], [148, 224, 224],
	[115, 194, 201], [84, 181, 181], [59, 158, 158], [36, 143, 143],
	[10, 125, 140], [0, 105, 133], [192, 192, 192], [255, 217, 143],
	[166, 117, 115], [102, 128, 128], [158, 99, 181], [212, 122, 0],
	[148, 0, 148], [66, 158, 176], [87, 23, 143], [0, 201, 0],
	[112, 212, 255], [255, 255, 199], [217, 255, 199], [199, 255, 199],
	[163, 255, 199], [143, 255, 199], [97, 255, 199], [69, 255, 199],
	[48, 255, 199], [31, 255, 199], [0, 255, 156], [0, 230, 117],
	[0, 212, 82], [0, 191, 56], [0, 171, 36], [77, 194, 255],
	[77, 166, 255], [33, 148, 214], [38, 125, 171], [38, 102, 150],
	[23, 84, 135], [208, 208, 224], [255, 209, 35], [184, 184, 208],
	[166, 84, 77], [87, 89, 97], [158, 79, 181], [171, 92, 0],
	[117, 79, 69], [66, 130, 150], [66, 0, 102], [0, 125, 0],
	[112, 171, 250], [0, 186, 255], [0, 161, 255], [0, 143, 255],
	[0, 128, 255], [0, 107, 255], [84, 92, 242], [120, 92, 227],
	[138, 79, 227], [161, 54, 212], [179, 31, 212], [179, 31, 186],
	[179, 13, 166], [189, 13, 135], [199, 0, 102], [204, 0, 89],
	[209, 0, 79], [217, 0, 69], [224, 0, 56], [230, 0, 46],
	[235, 0, 38]], start=1))


def element_color(element_number):
	rgb = _element_colors.get(element_number, [255, 0, 255])
	return rgb

# The following covalent bond radii are taken from: Elaine C. Meng
# and Richard A. Lewis, "Determination of Molecular Topology and
# Atomic Hybridization States from Heavy Atom Coordinates," Journal
# of Computational Chemistry, Vol. 12, No. 7, pp. 891-898 (1991).

_covalent = [
	0.00, 0.23, 0.00, 0.68,
	0.35, 0.83, 0.68, 0.68,
	0.68, 0.64, 0.00, 0.97,
	1.10, 1.35, 1.20, 1.05,
	1.02, 0.99, 0.00, 1.33,
	0.99, 1.44, 1.47, 1.33,
	1.35, 1.35, 1.34, 1.33,
	1.50, 1.52, 1.45, 1.22,
	1.17, 1.21, 1.22, 1.21,
	0.00, 1.47, 1.12, 1.78,
	1.56, 1.48, 1.47, 1.35,
	1.40, 1.45, 1.50, 1.59,
	1.69, 1.63, 1.46, 1.46,
	1.47, 1.40, 0.00, 1.67,
	1.34, 1.87, 1.83, 1.82,
	1.81, 1.80, 1.80, 1.99,
	1.79, 1.76, 1.75, 1.74,
	1.73, 1.72, 1.94, 1.72,
	1.57, 1.43, 1.37, 1.35,
	1.37, 1.32, 1.50, 1.50,
	1.70, 1.55, 1.54, 1.54,
	1.68, 0.00, 0.00, 0.00,
	1.90, 1.88, 1.79, 1.61,
	1.58, 1.55, 1.53, 1.51,
]

def element_radius(element_number):
	if element_number < 0:
		return 0.0
	try:
		return _covalent[element_number]
	except IndexError:
		return 0.0


def register():
	from chimera2 import io
	io.register_format("PDB", io.STRUCTURE,
		(".pdb", ".pdb1", ".ent", ".pqr"), ("pdb",),
		mime=("chemical/x-pdb", "chemical/x-spdbv"),
		reference="http://wwpdb.org/docs.html#format",
		open_func=open)
	io.register_format("CIF", io.STRUCTURE, (".cif"), ("cif", "cifID"),
		mime=("chemical/x-cif"),
		reference="http://www.iucr.org/__data/iucr/cif/standard/cifstd1.html")
	io.register_format("mmCIF", io.STRUCTURE, (".mcif"), ("mmcif",),
		mime=("chemical/x-mmcif"),
		reference="http://mmcif.rcsb.org/")
