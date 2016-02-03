// vi: set expandtab ts=4 sw=4:
#include <algorithm>
#include <set>

#include "Atom.h"
#include "Bond.h"
#include "destruct.h"
#include "Graph.h"

namespace atomstruct {

Graph::~Graph() {
    // need to assign to variable make it live to end of destructor
    auto du = DestructionUser(this);
    for (auto b: _bonds)
        delete b;
    for (auto a: _atoms)
        delete a;
}
    
void
Graph::delete_bond(Bond *b)
{
    typename Bonds::iterator i = std::find_if(_bonds.begin(), _bonds.end(),
        [&b](Bond* ub) { return ub == b; });
    if (i == _bonds.end())
        throw std::invalid_argument("delete_bond called for Bond not in Graph");
    auto db = DestructionBatcher(this);
    for (auto a: b->atoms())
        a->remove_bond(b);
    _bonds.erase(i);
    set_gc_shape();
    delete b;
}

void
Graph::delete_atom(Atom *a)
{
    typename Atoms::iterator i = std::find_if(_atoms.begin(), _atoms.end(),
        [&a](Atom* ua) { return ua == a; });
    if (i == _atoms.end())
        throw std::invalid_argument("delete_atom called for Atom not in Graph");
    auto db = DestructionBatcher(this);
    for (auto b: a->bonds())
        b->other_atom(a)->remove_bond(b);
    _atoms.erase(i);
    set_gc_shape();
    delete a;
}

void
Graph::delete_atoms(const std::set<Atom*>& atoms)
{
    auto db = DestructionBatcher(this);
    // remove_if doesn't swap the removed items into the end of the vector,
    // so can't just go through the tail of the vector and delete things,
    // need to delete them as part of the lambda
    auto new_a_end = std::remove_if(_atoms.begin(), _atoms.end(),
        [&atoms](Atom* a) { 
            bool rm = atoms.find(a) != atoms.end();
            if (rm) delete a; return rm;
        });
    _atoms.erase(new_a_end, _atoms.end());

    for (auto a: _atoms) {
        std::vector<Bond*> removals;
        for (auto b: a->bonds()) {
            if (atoms.find(b->other_atom(a)) != atoms.end())
                removals.push_back(b);
        }
        for (auto b: removals)
            a->remove_bond(b);
    }

    auto new_b_end = std::remove_if(_bonds.begin(), _bonds.end(),
        [&atoms](Bond* b) {
            bool rm = atoms.find(b->atoms()[0]) != atoms.end()
            || atoms.find(b->atoms()[1]) != atoms.end();
            if (rm) delete b; return rm;
        });
    _bonds.erase(new_b_end, _bonds.end());
    set_gc_shape();
}

void
Graph::set_color(const Rgba& rgba)
{
    for (auto a: _atoms)
        a->set_color(rgba);
    for (auto b: _bonds)
        b->set_color(rgba);
}

} //  namespace atomstruct
