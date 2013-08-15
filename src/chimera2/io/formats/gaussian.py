def register():
	from chimera2 import io
	io.register_format("Gaussian formatted checkpoint", io.STRUCTURE, 
		(".fchk",), ("fchk", "gaussian"),
		mime=("chemical/x-gaussian-checkpoint",),
		reference="http://www.gaussian.com/g_tech/g_ur/f_formchk.htm")
