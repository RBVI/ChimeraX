TOP = ../..
include $(TOP)/mk/config.make

all:

install:
	echo 'for now: "make run" to run qt-based chimera2'

run:
	# by running within make, the right environment variables are setup
	./chimera2

run-python:
	# by running within make, the right environment variables are setup
	$(bindir)/python3

nogui:
	# by running within make, the right environment variables are setup
	./chimera2 --nogui -d json

designer:
	$(bindir)/designer main.ui

gdb:
	gdb $(bindir)/python3 core

clean:
	rm -rf __pycache__
