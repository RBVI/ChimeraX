PYTHON = ../ChimeraX.app/Contents/bin/python3.8

wheel: chimerax
	$(PYTHON) setup.py bdist_wheel

chimerax:
	ln -s ../ChimeraX.app/Contents/lib/python3.8/site-packages/chimerax .

clean:
	\rm -rf dist build chimerax.egg-info
	\rm chimerax

