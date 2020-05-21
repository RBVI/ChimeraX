#pragma once

// Main entry point to ChimeraX C++ API for working with atomic models
#include <atomstruct/AtomicStructure.h>

#include <tuple>

static char _sample_module_doc[] = "Sample support module";

// Will become the docstring for the module's repr() method
static char _sample_repr_doc[] =
"repr(o)\n"
"\n"
"Return string representation of o.";

// Will become the docstring for the "counts" method
static char _sample_counts_doc[] =
"repr(o)\n"
"\n"
"Return 2-tuple of number of atoms and bonds in a structure.";

std::tuple<size_t, size_t> atom_and_bond_count(void *structure_pointer)
{
    // Pointers are all held in Python as void, so we have to cast to the
    // required type
    atomstruct::Structure *s = static_cast<atomstruct::Structure *>(structure_pointer);
    return std::make_tuple(s->num_atoms(), s->num_bonds());
}
