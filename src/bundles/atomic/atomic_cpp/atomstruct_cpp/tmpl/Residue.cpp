// vi: set expandtab ts=4 sw=4:

/*
 * === UCSF ChimeraX Copyright ===
 * Copyright 2016 Regents of the University of California.
 * All rights reserved.  This software provided pursuant to a
 * license agreement containing restrictions on its disclosure,
 * duplication and use.  For details see:
 * http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html
 * This notice must be embedded in or attached to all copies,
 * including partial copies, of the software or any revisions
 * or derivations thereof.
 * === UCSF ChimeraX Copyright ===
 */

#define ATOMSTRUCT_EXPORT
#define PYINSTANCE_EXPORT
#include "restmpl.h"

#include "TemplateCache.h"

#include <pyinstance/PythonInstance.instantiate.h>
template class pyinstance::PythonInstance<tmpl::Residue>;

namespace tmpl {

#if 0
std::vector<Atom *>
Residue::template_assign(void (Atom::*assign_func)(const char *),
    const char *app, const char *template_dir, const char *extension) const
{
    return template_assign(tmpl_assigner(assign_func),
                        app, template_dir, extension);
}

std::vector<Atom *>
Residue::template_assign(void (*assign_func)(Atom *, const char *),
    const char *app, const char *template_dir, const char *extension) const
{
    return template_assign(tmpl_assigner(assign_func),
                        app, template_dir, extension);
}
#endif

// returns atoms that received assignments from the template.
// can throw exceptions if:
//     template syntax error: TA_TemplateSyntax
//    no template found: TA_NoTemplate
//    internal logic error: std::logic_error
std::vector<Atom *>
Residue::template_assign(void (Atom::*assign)(const AtomType&),
    const char *app, const char *template_dir, const char *extension) const
{
    TemplateCache *tc = TemplateCache::template_cache();
    TemplateCache::AtomMap *am = tc->res_template(name(), app,
                            template_dir, extension);
    std::vector<Atom *> assigned;
    for (AtomsMap::const_iterator ai = _atoms.begin(); ai != _atoms.end(); ++ai) {
        const AtomName& at_name = ai->first;
        Atom *a = ai->second;

        TemplateCache::AtomMap::iterator ami = am->find(at_name);
        if (ami == am->end())
            continue;
        
        AtomType norm_type(ami->second.first);
        ConditionalTemplate *ct = ami->second.second;
        if (ct != NULL) {
            // assign conditional type if applicable
            bool cond_assigned = false;
            for (std::vector<CondInfo>::iterator cii =
            ct->conditions.begin(); cii != ct->conditions.end();
            ++cii) {
                CondInfo &ci = *cii;
                if (ci.op == ".") {
                      // is given atom terminal?
                    bool is_terminal = true;
                    AtomsMap::const_iterator opai =
                        _atoms.find(ci.operand.c_str());
                    if (opai == _atoms.end())
                        continue;
                    Atom *opa = opai->second;
                    for (auto bonded: opa->neighbors()) {
                        if (bonded->residue() != opa->residue()) {
                            is_terminal = false;
                            break;
                        }
                    }
                    if (is_terminal) {
                        cond_assigned = true;
                        if (ci.result != "-") {
                            (a->*assign)(ci.result);
                            assigned.push_back(a);
                        }
                    }
                } else if (ci.op == "?") {
                      // does given atom exist in residue?
                    if (_atoms.find(ci.operand.c_str()) != _atoms.end()) {
                        cond_assigned = true;
                        if (ci.result != "-") {
                            (a->*assign)(ci.result);
                            assigned.push_back(a);
                        }
                    }
                } else {
                      throw std::logic_error(
                    "Legal template condition not implemented");
                }
                if (cond_assigned)
                    break;
            }
            if (cond_assigned)
                continue;
        }

        // assign normal type
        if (norm_type != "-") {
            (a->*assign)(norm_type);
            assigned.push_back(a);
        }
    }
    return assigned;
}

std::vector<Atom*>
Residue::atoms() const
{
    std::vector<Atom*> atoms;
    for (auto name_atom: atoms_map()) {
        atoms.push_back(name_atom.second);
    }
    return atoms;
}

void
Residue::add_atom(Atom *atom)
{
    atom->_residue = this;
    _atoms[atom->name()] = atom;
    _has_metal = _has_metal || atom->element().is_metal();
}

Atom *
Residue::find_atom(const AtomName& index) const
{
    AtomsMap::const_iterator i = _atoms.find(index);
    if (i == _atoms.end())
        return NULL;
    return i->second;
}

void
Residue::add_link_atom(Atom *element)
{
    if (_link_atoms.empty())
        _link_atoms.reserve(2);
    _link_atoms.push_back(element);
}

Residue::Residue(Molecule *, const char *n): pdbx_ambiguous(false), _name(n), _chief(0), _link(0), _has_metal(false)
{
}

Residue::~Residue()
{
}

}  // namespace tmpl
