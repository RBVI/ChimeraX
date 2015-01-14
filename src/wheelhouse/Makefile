TOP = ../..
include $(TOP)/mk/config.make

PYSRCS = update_metadata.py

all: $(PYOBJS)

install: dir
	$(RSYNC) $(PYSRCS) $(wheelhouse)
	$(RSYNC) Makefile.wheelhouse $(wheelhouse)/Makefile

dir:
	if [ ! -d "$(wheelhouse)" ]; then mkdir -p $(wheelhouse); fi

lint:
	$(PYLINT) $(PYSRCS)

clean:
	rm -rf __pycache__
