include ../Makefile.bundle

DEVEL_ARGS = test false

wheel install app-install:	lib

lib:
	$(MAKE) -C _graphics install

clean:	_graphics.clean

_graphics.clean:
	$(MAKE) -C _graphics clean
