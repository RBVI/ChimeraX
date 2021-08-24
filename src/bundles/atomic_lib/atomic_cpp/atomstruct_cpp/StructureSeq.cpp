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

#include <algorithm>
#include <exception>
#include <Python.h>
#include <sstream>

#define ATOMSTRUCT_EXPORT
#define PYINSTANCE_EXPORT
#include "Chain.h"
#include "ChangeTracker.h"
#include "destruct.h"
#include "Structure.h"
#include "Sequence.h"
#include "Residue.h"

namespace atomstruct {

StructureSeq::StructureSeq(const ChainID& chain_id, Structure* s, PolymerType pt):
    Sequence(std::string("chain ") + (chain_id == " " ? "(blank)" : chain_id.c_str())),
    _chain_id(chain_id), _from_seqres(false), _polymer_type(pt), _structure(s)
{ }

void
StructureSeq::bulk_set(const StructureSeq::Residues& residues,
    const Sequence::Contents* chars)
{
    bool del_chars = chars == nullptr;
    if (del_chars) {
        auto res_chars = new Sequence::Contents(residues.size());
        auto chars_ptr = res_chars->begin();
        for (auto ri = residues.begin(); ri != residues.end();
        ++ri, ++chars_ptr) {
            (*chars_ptr) = Sequence::rname3to1((*ri)->name());
        }
        chars = res_chars;
    }
    _residues = residues;
    assign(chars->begin(), chars->end());
    _res_map.clear();
    bool ischain = is_chain();
    int i = 0;
    for (auto ri = residues.begin(); ri != residues.end(); ++ri, ++i) {
        if (*ri != nullptr) {
            _res_map[*ri] = i;
            if (ischain)
                (*ri)->set_chain(dynamic_cast<Chain*>(this));
        }
    }

    if (del_chars)
        delete chars;
    if (ischain)
        _structure->change_tracker()->add_modified(structure(), dynamic_cast<Chain*>(this),
            ChangeTracker::REASON_SEQUENCE, ChangeTracker::REASON_RESIDUES);
}

void
StructureSeq::clear_residues() {
    // only called from ~AtomicStructure...
    _residues.clear();
    _res_map.clear();
    _structure->change_tracker()->add_modified(_structure, dynamic_cast<Chain*>(this),
        ChangeTracker::REASON_RESIDUES);
    demote_to_sequence();
}

StructureSeq*
StructureSeq::copy() const
{
    StructureSeq* ss = new StructureSeq(_chain_id, _structure);
    ss->bulk_set(_residues, &_contents);
    return ss;
}

void
StructureSeq::demote_to_sequence()
{
    if (is_chain()) {
        _structure->change_tracker()->add_deleted(_structure, dynamic_cast<Chain*>(this));
    }
    _structure = nullptr;
    Py_XDECREF(py_call_method("_cpp_demotion"));
    // let normal deletion processes clean up; don't explicitly delete here
}

void
StructureSeq::destructors_done(const std::set<void*>& destroyed)
{
    if (is_chain())
        // Chains keep their residue lists up to date "by hand"
        return;
    // StructureSeq has to keep its residue list up to date itself
    std::set<Residue*> destroyed_residues;
    for (auto res_pos: _res_map)
        if (destroyed.find(static_cast<void*>(res_pos.first)) != destroyed.end())
            destroyed_residues.insert(res_pos.first);
    if (destroyed_residues.size() > 0)
        remove_residues(destroyed_residues);
}

void
StructureSeq::insert(Residue* follower, Residue* r)
{
    auto ri = std::find(_residues.begin(), _residues.end(), follower);
    if (ri == _residues.end())
        throw std::logic_error("insert-before residue not found in _residues");

    if (r->chain() != nullptr)
        r->chain()->remove_residue(r);
    Sequence::insert(Sequence::begin() + (ri - _residues.begin()), 1, Sequence::rname3to1(r->name()));
    _res_map[r] = _res_map[follower];
    for (auto ri2 = ri; ri2 != _residues.end(); ++ri2)
        _res_map[*ri2]++;
    _residues.insert(ri, r);
    if (is_chain()) {
        r->set_chain(dynamic_cast<Chain*>(this));
        _structure->change_tracker()->add_modified(_structure, dynamic_cast<Chain*>(this),
            ChangeTracker::REASON_SEQUENCE, ChangeTracker::REASON_RESIDUES);
    }
}

StructureSeq&
StructureSeq::operator+=(StructureSeq& addition)
{
    Sequence::operator+=(*this);
    auto offset = _residues.size();
    _residues.insert(_residues.end(), addition._residues.begin(), addition._residues.end());
    bool ischain = is_chain();
    for (auto res_i: addition._res_map) {
        _res_map[res_i.first] = res_i.second + offset;
        if (ischain)
            // assuming we're not calling remove_residue() later, which would
            // null out the chain pointer...
            res_i.first->set_chain(dynamic_cast<Chain*>(this));
    }
    if (ischain) {
        _structure->remove_chain(dynamic_cast<Chain*>(&addition));
        addition.demote_to_sequence();
        _structure->change_tracker()->add_modified(_structure, dynamic_cast<Chain*>(this),
            ChangeTracker::REASON_SEQUENCE, ChangeTracker::REASON_RESIDUES);
    }
    return *this;
}

void
StructureSeq::pop_back()
{
    Sequence::pop_back();
    _residues.pop_back();
    auto back = _residues.back();
    if (back != nullptr) {
        _res_map.erase(back);
        // demote_to_sequence sets _structure to null, so...
        auto structure = _structure;
        bool ischain = is_chain();
        if (no_structure_left()) {
            if (ischain)
                structure->remove_chain(dynamic_cast<Chain*>(this));
            demote_to_sequence();
        }
        if (ischain) {
            back->set_chain(nullptr);
            structure->change_tracker()->add_modified(structure, dynamic_cast<Chain*>(this),
                ChangeTracker::REASON_SEQUENCE, ChangeTracker::REASON_RESIDUES);
        }
    }
}

void
StructureSeq::pop_front()
{
    Sequence::pop_front();
    auto front = _residues.front();
    _residues.erase(_residues.begin());
    if (front != nullptr) {
        _res_map.erase(front);
        bool ischain = is_chain();
        for (auto& res_i: _res_map)
            res_i.second--;
        // demote_to_sequence sets _structure to null, so...
        auto structure = _structure;
        if (no_structure_left()) {
            if (ischain)
                structure->remove_chain(dynamic_cast<Chain*>(this));
            demote_to_sequence();
        }
        if (ischain) {
            front->set_chain(nullptr);
            structure->change_tracker()->add_modified(structure, dynamic_cast<Chain*>(this),
                ChangeTracker::REASON_SEQUENCE, ChangeTracker::REASON_RESIDUES);
        }
    }
}

void
StructureSeq::push_back(Residue* r)
{
    if (r->chain() != nullptr)
        r->chain()->remove_residue(r);
    Sequence::push_back(Sequence::rname3to1(r->name()));
    _res_map[r] = _residues.size();
    _residues.push_back(r);
    if (is_chain()) {
        r->set_chain(dynamic_cast<Chain*>(this));
        _structure->change_tracker()->add_modified(_structure, dynamic_cast<Chain*>(this),
            ChangeTracker::REASON_SEQUENCE, ChangeTracker::REASON_RESIDUES);
    }
}

void
StructureSeq::push_front(Residue* r)
{
    if (r->chain() != nullptr)
        r->chain()->remove_residue(r);
    Sequence::push_front(Sequence::rname3to1(r->name()));
    Residues pushed;
    pushed.reserve(_residues.size()+1);
    pushed.push_back(r);
    pushed.insert(pushed.end(), _residues.begin(), _residues.end());
    pushed.swap(_residues);
    for (auto& res_i: _res_map) {
        res_i.second++;
    }
    _res_map[r] = 0;
    if (is_chain()) {
        r->set_chain(dynamic_cast<Chain*>(this));
        _structure->change_tracker()->add_modified(_structure, dynamic_cast<Chain*>(this),
            ChangeTracker::REASON_SEQUENCE, ChangeTracker::REASON_RESIDUES);
    }
}

void
StructureSeq::remove_residue(Residue* r) {
    std::set<Residue*> residues;
    residues.insert(r);
    remove_residues(residues);
}

void
StructureSeq::remove_residues(std::set<Residue*>& residues) {
    // Chain getting demoted to sequence will let destructors_done() to call
    // again for the same set of residues, so prevent shenanigans...
    if (_res_map.size() == 0)
        return;
    bool ischain = is_chain();
    for (auto r: residues) {
        auto ri = std::find(_residues.begin(), _residues.end(), r);
        *ri = nullptr;
        if (ischain)
            r->set_chain(nullptr);
    }
    if (ischain)
        _structure->change_tracker()->add_modified(_structure, dynamic_cast<Chain*>(this),
            ChangeTracker::REASON_RESIDUES);
    if (_res_map.size() == residues.size()) {
        _res_map.clear();
        if (ischain) {
            if (DestructionCoordinator::destruction_parent() != _structure)
                _structure->remove_chain(dynamic_cast<Chain*>(this));
        }
        demote_to_sequence();
    } else {
        _res_map.clear();
        int i = 0;
        for (auto ri = _residues.begin(); ri != _residues.end(); ++ri, ++i) {
            if (*ri != nullptr) {
                _res_map[*ri] = i;
            }
        }
        Py_XDECREF(py_call_method("_cpp_modified"));
    }
}

void
StructureSeq::session_restore(int version, int** ints, float** floats)
{
    Sequence::session_restore(version, ints, floats);

    auto& int_ptr = *ints;
    _from_seqres = int_ptr[0];
    auto res_map_size = int_ptr[1];
    auto residues_size = int_ptr[2];
    if (version >= 10) {
        _polymer_type = static_cast<PolymerType>(int_ptr[3]);
    }
    int_ptr += SESSION_NUM_INTS(version);

    auto& residues = _structure->residues();
    for (decltype(res_map_size) i = 0; i < res_map_size; ++i) {
        auto res_index = *int_ptr++;
        auto pos = *int_ptr++;
        auto res = residues[res_index];
        _res_map[res] = pos;
        res->set_chain(dynamic_cast<Chain*>(this));
        if (i == 0 && version < 10) {
            // polymer type stored in residues
            _polymer_type = Sequence::rname_polymer_type(res->name());
        }
    }

    _residues.reserve(residues_size);
    for (decltype(residues_size) i = 0; i < residues_size; ++i) {
        auto res_index = *int_ptr++;
        if (res_index < 0)
            _residues.push_back(nullptr);
        else
            _residues.push_back(residues[res_index]);
    }
}

void
StructureSeq::session_save(int** ints, float** floats) const
{
    Sequence::session_save(ints, floats);

    auto& int_ptr = *ints;
    int_ptr[0] = _from_seqres;
    int_ptr[1] = _res_map.size();
    int_ptr[2] = _residues.size();
    int_ptr[3] = (int)_polymer_type;
    int_ptr += SESSION_NUM_INTS();

    auto& ses_res = *_structure->session_save_residues;
    for (auto r_pos: _res_map) {
        *int_ptr++ = ses_res[r_pos.first];
        *int_ptr++ = r_pos.second;
    }
    for (auto r: _residues) {
        if (r == nullptr)
            *int_ptr++ = -1;
        else
            *int_ptr++ = ses_res[r];
    }
}

void 
StructureSeq::set(unsigned i, Residue *r, char character)
{
    unsigned char c;
    if (character < 0) {
        c = Sequence::rname3to1(r->name());
    } else {
        c = (unsigned char)character;
    }
    bool ischain = is_chain();
    if (i == _residues.size()) {
        _residues.push_back(r);
        Sequence::push_back(c);
    } else {
        auto& res_at_i = _residues.at(i);
        if (res_at_i != nullptr) {
            _res_map.erase(res_at_i);
            if (ischain)
                res_at_i->set_chain(nullptr);
        }
        res_at_i = r;
        at(i) = c;
    }
    // demote_to_sequence sets _structure to null, so...
    auto structure = _structure;
    if (r != nullptr) {
        _res_map[r] = i;
        if (ischain)
            r->set_chain(dynamic_cast<Chain*>(this));
    } else {
        if (no_structure_left()) {
            if (ischain)
                structure->remove_chain(dynamic_cast<Chain*>(this));
            demote_to_sequence();
        }
    }
    if (ischain)
        structure->change_tracker()->add_modified(structure, dynamic_cast<Chain*>(this),
            ChangeTracker::REASON_SEQUENCE, ChangeTracker::REASON_RESIDUES);
}

void
StructureSeq::set_chain_id(ChainID chain_id)
{
    if (chain_id != _chain_id) {
        _chain_id = chain_id;
        if (is_chain()) {
            _structure->change_tracker()->add_modified(_structure, dynamic_cast<Chain*>(this),
                ChangeTracker::REASON_CHAIN_ID);
            for (auto r: residues())
                if (r != nullptr)
                    _structure->change_tracker()->add_modified(_structure, r,
                        ChangeTracker::REASON_CHAIN_ID);
        }
    }
}

void
StructureSeq::set_from_seqres(bool fs)
{
    if (fs == _from_seqres)
        return;
    if (_from_seqres) {
        // changing from true to false;
        // eliminate seqres parts of sequence...
        if (std::find(_residues.begin(), _residues.end(), nullptr)
        != _residues.end()) {
            // there actually are seqres portions
            _res_map.clear();
            StructureSeq::Residues new_residues;
            Sequence::Contents new_contents;
            auto ri = _residues.begin();
            int i = 0;
            for (auto si = begin(); si != end(); ++si, ++ri) {
                if (*ri == nullptr)
                    continue;
                _res_map[*ri] = ++i;
                new_residues.push_back(*ri);
                new_contents.push_back(*si);
            }
            _residues.swap(new_residues);
            swap(new_contents);
        }
    }
    _from_seqres = fs;
    if (is_chain())
        _structure->change_tracker()->add_modified(_structure, dynamic_cast<Chain*>(this),
            ChangeTracker::REASON_SEQUENCE, ChangeTracker::REASON_RESIDUES);
}

}  // namespace atomstruct
