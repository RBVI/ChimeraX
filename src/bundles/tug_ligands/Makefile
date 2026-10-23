include ../Makefile.bundle
include $(TOP)/mk/config.make

# Get OpenMM Amber parameters for ligands, originally from the ISOLDE github repository
#   https://github.com/tristanic/isolde/tree/master/isolde/src/openmm/amberff
DISTCLEAN = src/gaff2.xml src/moriarty_and_case_ligands.zip

wheel install install-app: src/gaff2.xml src/moriarty_and_case_ligands.zip

src/gaff2.xml:
	$(call FETCH_PREREQ_TO,$@,$(PREREQS_ARCHIVE)/tug_ligands/gaff2.xml)

src/moriarty_and_case_ligands.zip:
	$(call FETCH_PREREQ_TO,$@,$(PREREQS_ARCHIVE)/tug_ligands/moriarty_and_case_ligands.zip)
