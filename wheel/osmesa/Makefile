PYTHON = ../../ChimeraX.app/Contents/bin/python3.9

wheel: osmesa
	$(PYTHON) setup.py bdist_wheel

osmesa:
	ln -s ../../ChimeraX.app/Contents/lib/python3.9/site-packages/osmesa .

clean:
	\rm -rf dist build osmesa.egg-info
	\rm osmesa

