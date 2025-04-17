This folder prepares a wheel for distribution on PyPi, derived from a built ChimeraX.

The chimerax folder from $(APP_PYSITEDIR) is copied into this folder for preprocessing
before that wheel is built. 

Not all of ChimeraX is distributed in the wheel. We strive to keep GUI modules out of 
the wheel. Excluded files and modules can be found in filter_modules.py, along with 
the reasons for excluding them. This list of excluded modules are read at build-time 
and deleted from the ChimeraX folder. They are also read in the import tests in 
tests/core/test_all_imports.py.

Some of our dependencies exist for the GUI or for building. Those are not listed as 
dependencies for the wheel. Excluded dependencies can be found in filter_dependencies.py

Finally, the wheel version is taken from the built ChimeraX version. 

The point of all this is that there should be a single source of truth for each aspect
of the wheel; if it's not calculated from ChimeraX, it should be calculable from metadata
(prereqs/pips) somehow. We do not want this folder to diverge from ChimeraX to the point
that it becomes a significant maintenance burden.
