TEST = t2

OPT = -g
WRAPPY = ../wrappy2

#CXX	= cxx -std strict_ansi -model ansi -msg_disable nostdlonglong -w0
#CXXFLAGS= $(OPT) -I/usr/local/include/python2.4 -I. -I$(OTF)/include
#CXX	= CC -LANG:std
#CXXFLAGS= $(OPT) -I/usr/local/src/staff/gregc/chimera2-IRIX/build/include/python2.4 -I.
CXX	= gcc -pipe -pthread -fPIC -Wall -Wno-invalid-offsetof
CXXFLAGS= $(OPT) -I$(OTF)/include/python2.5 -I$(OTF)/include

module.so: module.o xyzzy_Object.o $(TEST).o
	$(CXX) $(OPT) -shared -o module.so module.o xyzzy_Object.o $(TEST).o -L../lib -lwrappy2 -lpython2.4

module.cpp xyzzy_Object.cpp: $(TEST).h
	$(WRAPPY) -N plugh $(TEST).h

clean:
	rm -f module.* xyzzy_Object.* $(TEST).o so_locations core errs
	rm -rf cxx_repository ii_files

distclean: clean
