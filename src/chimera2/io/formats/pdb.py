def register():
	from chimera2 import io
	io.register_format("PDB", io.STRUCTURE,
		(".pdb", ".pdb1", ".ent", ".pqr"), ("pdb",),
		mime=("chemical/x-pdb", "chemical/x-spdbv"),
		reference="http://wwpdb.org/docs.html#format")
	io.register_format("CIF", io.STRUCTURE, (".cif"), ("cif", "cifID"),
		mime=("chemical/x-cif"),
		reference="http://www.iucr.org/__data/iucr/cif/standard/cifstd1.html")
	io.register_format("mmCIF", io.STRUCTURE, (".mcif"), ("mmcif",),
		mime=("chemical/x-mmcif"),
		reference="http://mmcif.rcsb.org/")
