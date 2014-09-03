TOP = ../..
include $(TOP)/mk/config.make

SUPPRESSIONS=$(wildcard $(tmpdir)/Python-*/Misc/valgrind-python.supp)
VALGRIND = valgrind --tool=memcheck --error-limit=no --track-origins=yes \
	   --num-callers=30 --gen-suppressions=all \
	   --suppressions=$(SUPPRESSIONS)

all:

install:
	echo 'for now: "make run" to run qt-based chimera2'

run:
	# by running within make, the right environment variables are setup
	./chimera2

profile:
	echo "Decorate functions with @profile."
	echo "Use 'make profile_results' to see results."
	./chimera2 --profile

profile-results:
	python3 -m line_profiler chimera2.lprof

run-python:
	# by running within make, the right environment variables are setup
	$(bindir)/python3

run-valgrind:
	@echo "==== Edit suppressions file to suppress more Python errors ===="
	$(VALGRIND) $(bindir)/python3 chimera2_main.py

nogui:
	# by running within make, the right environment variables are setup
	./chimera2 --nogui -d json

designer:
	$(bindir)/designer main.ui

gdb:
	env PYTHONPATH=$(libdir)/python3.4 gdb $(bindir)/python3 core

clean:
	rm -rf __pycache__
