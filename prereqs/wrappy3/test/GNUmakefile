TEST = t2

OPT = -g
WRAPPY = ../wrappy3

CXX	= gcc -pipe -pthread -fPIC -Wall -Wno-invalid-offsetof
CXXFLAGS= $(OPT) -I$(OTF)/include/python3.5m -I$(OTF)/include

module.so: module.o xyzzy_Object.o $(TEST).o
	$(CXX) $(OPT) -shared -o module.so module.o xyzzy_Object.o $(TEST).o -L../lib -lwrappy2 -lpython2.4

module.cpp xyzzy_Object.cpp: $(TEST).h
	$(WRAPPY) -N plugh $(TEST).h

clean:
	rm -f module.* xyzzy_Object.* $(TEST).o so_locations core errs
	rm -rf cxx_repository ii_files

distclean: clean
