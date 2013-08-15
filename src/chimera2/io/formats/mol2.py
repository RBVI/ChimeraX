def register():
	from chimera2 import io
	io.register_format("Mol2", io.STRUCTURE,
		(".mol2",), ("mol2",), mime=("chemical/x-mol2",),
		reference="http://www.tripos.com/tripos_resources/fileroot/pdfs/mol2_format.pdf")
