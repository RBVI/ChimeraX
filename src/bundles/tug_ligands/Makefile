include ../Makefile.bundle
include $(TOP)/mk/config.make

# Get OpenMM Amber parameters for ligands, originally from the ISOLDE github repository
#   https://github.com/tristanic/isolde/tree/master/isolde/src/openmm/amberff

wheel install install-app: src/gaff2.xml src/moriarty_and_case_ligands.zip

src/gaff2.xml:
	cd src ; $(FETCH_PREREQ) $(PREREQS_ARCHIVE)/tug_ligands/gaff2.xml

src/moriarty_and_case_ligands.zip:
	cd src ; $(FETCH_PREREQ) $(PREREQS_ARCHIVE)/tug_ligands/moriarty_and_case_ligands.zip
