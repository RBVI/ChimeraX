TOP = ../..
include $(TOP)/mk/config.make

all:

install:
	echo 'for now: "make run" to run qt-based chimera2'

run:
	# by running within make, the right environment variables are setup
	./chimera2

nogui:
	# by running within make, the right environment variables are setup
	./chimera2 --nogui -d json

clean:
	rm -f *.pyc
