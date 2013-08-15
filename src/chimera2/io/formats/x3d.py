def register():
	from chimera2 import io
	io.register_format("VRML", io.GENERIC3D,
		(".wrl", ".vrml"), ("vrml",), mime=("model/vrml",),
		reference="http://www.web3d.org/x3d/specifications/#vrml97")
	io.register_format("X3D", io.GENERIC3D,
		(".x3d",), ("x3d",), mime=("model/x3d+xml",),
		reference="http://www.web3d.org/x3d/specifications/#x3d-spec")
