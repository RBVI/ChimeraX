include ../Makefile.bundle

CLEAN_ALWAYS += src/dcd/MDToolsMarch97/__pycache__ \
		src/dcd/__pycache \
		src/_gromacs*.$(PYMOD_EXT)

wheel install app-install uv-wheel uv-install:	xdrfile_lib

xdrfile_lib:
	$(MAKE) -C gromacs all

clean:	gromacs_clean

gromacs_clean:
	$(MAKE) -C gromacs clean
