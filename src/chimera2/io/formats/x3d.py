from chimera2 import generic3d

def register():
	from chimera2 import io
	io.register_format("VRML", generic3d.CATEGORY,
		(".wrl", ".vrml"), ("vrml",), mime=("model/vrml",),
		reference="http://www.web3d.org/x3d/specifications/#vrml97")
	io.register_format("X3D", generic3d.CATEGORY,
		(".x3d",), ("x3d",), mime=("model/x3d+xml",),
		reference="http://www.web3d.org/x3d/specifications/#x3d-spec")
