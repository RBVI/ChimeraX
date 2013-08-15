def register():
	from chimera2 import io
	io.register_format("Python", io.SCRIPT,
		(".py", ".pyc", ".pyo", ".pyw"), ("python", "py", "chimera"),
		mime=("application/x-chimera",),
		reference="http://www.python.org/")
