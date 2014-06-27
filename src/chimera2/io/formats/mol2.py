from chimera2 import molecule

def register():
	from chimera2 import io
	io.register_format("Mol2", molecule.CATEGORY,
		(".mol2",), ("mol2",), mime=("chemical/x-mol2",),
		reference="http://www.tripos.com/tripos_resources/fileroot/pdfs/mol2_format.pdf")
