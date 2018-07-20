# The "make" targets are:
# 	wheel: build a Python wheel in "dist" directory.
# 	install: build wheel (if needed) and install in ChimeraX.
# 	test: run ChimeraX
# 	debug: run ChimeraX with debugging flag set
# 	clean: remove files used in building wheel

CHIMERAX_APP = ../../../ChimeraX.app

# Platform-dependent settings.  Should not need fixing.
# For Windows, we assume Cygwin is being used.
OS = $(patsubst CYGWIN_NT%,CYGWIN_NT,$(shell uname -s))
ifeq ($(OS),CYGWIN_NT)
CHIMERAX_EXE = $(CHIMERAX_APP)/bin/ChimeraX.exe
endif
ifeq ($(OS),Darwin)
CHIMERAX_EXE = $(CHIMERAX_APP)/Contents/bin/ChimeraX
endif
ifeq ($(OS),Linux)
CHIMERAX_EXE = $(CHIMERAX_APP)/bin/ChimeraX
endif
RUN = $(CHIMERAX_EXE) --nogui --exit --cmd

SRCS = $(wildcard src/*.py)
LIBS = src/lib/flot src/lib/rateyo src/lib/jquery.multiselect.js

wheel:	bundle_info.xml $(SRCS) $(LIBS)
	$(RUN) "devel build . exit true"

install app-install:	bundle_info.xml $(SRCS) $(LIBS)
	$(RUN) "devel install . user false exit true"

test:
	for t in $(wildcard test*.cxc) $(wildcard test*.py);\
		do $(CHIMERAX_EXE) --exit --nogui $$t;\
	done

debug:
	$(CHIMERAX_EXE) --debug

clean:
	if [ -x $(CHIMERAX_EXE) ]; then \
		$(RUN) "devel clean . exit true" ; \
	else \
		rm -rf build dist *.egg-info src/__pycache__ ; \
	fi

pylint:
	$(CHIMERAX_EXE) -m flake8 $(filter %.py, $(SRCS))

# From http://www.flotcharts.org/
src/lib/flot:	flot-0.8.3.zip
	# touch needed because unzip restores update time
	rm -rf $@ && unzip $? && mv flot $@ && touch $@

# From http://rateyo.fundoocode.ninja/
src/lib/rateyo:	rateyo-2.3.2.zip
	rm -rf $@ && unzip $? && mv v2.3.2 $@ && rm -rf __MACOSX && touch $@

# From https://www.jqueryscript.net/form/jQuery-Plugin-For-Multi-Select-List-with-Checkboxes-MultiSelect.html
MS_DIR = jQuery-Plugin-For-Multi-Select-List-with-Checkboxes-MultiSelect
src/lib/jquery.multiselect.js:	multiselect.zip
	unzip $? && mv $(MS_DIR)/jquery.multiselect.* src/lib \
		&& rm -rf $(MS_DIR) && touch $@
