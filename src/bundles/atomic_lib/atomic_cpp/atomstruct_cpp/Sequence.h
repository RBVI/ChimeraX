// vi: set expandtab ts=4 sw=4:

/*
 * === UCSF ChimeraX Copyright ===
 * Copyright 2022 Regents of the University of California. All rights reserved.
 * This software is provided pursuant to the ChimeraX license agreement, which
 * covers academic and commercial uses. For more information, see
 * <http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
 *
 * This file is part of the ChimeraX library. You can also redistribute and/or
 * modify it under the GNU Lesser General Public License version 2.1 as
 * published by the Free Software Foundation. For more details, see
 * <https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html>
 *
 * This file is distributed WITHOUT ANY WARRANTY; without even the implied
 * warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. This notice
 * must be embedded in or attached to all copies, including partial copies, of
 * the software or any revisions or derivations thereof.
 * === UCSF ChimeraX Copyright ===
 */

#ifndef atomstruct_Sequence
#define atomstruct_Sequence

#include <cctype>
#include <map>
#include <pyinstance/PythonInstance.declare.h>
#include <stdexcept>
#include <string>
#include <vector>

#include "imex.h"
#include "polymer.h"
#include "session.h"
#include "string_types.h"

// "forward declare" PyObject, which is a typedef of a struct,
// as per the python mailing list:
// http://mail.python.org/pipermail/python-dev/2003-August/037601.html
#ifndef PyObject_HEAD
struct _object;
typedef _object PyObject;
#endif

namespace atomstruct {

class SeqIndexError: public std::range_error {
public:
    SeqIndexError(const std::string& msg) : std::range_error(msg) {}
};

class ATOMSTRUCT_IMEX Sequence: public pyinstance::PythonInstance<Sequence> {
public:
    typedef std::vector<char>  Contents;
protected:
    typedef std::map<ResName, char>  _1Letter_Map;
    static void  _init_rname_map();
    static _1Letter_Map  _nucleic3to1;
    static _1Letter_Map  _protein3to1;
    static _1Letter_Map  _rname3to1;

    mutable std::map<unsigned int, unsigned int>  _cache_g2ug;
    mutable std::map<unsigned int, unsigned int>  _cache_ug2g;
    mutable Contents  _cache_ungapped;
    bool  _circular = false;
    void  _clear_cache() const
        { _cache_ungapped.clear();  _cache_g2ug.clear(); _cache_ug2g.clear(); }
    // can't inherit from vector, since we need to clear caches on changes
    Contents  _contents;
    std::string  _name;

private:
    static int  SESSION_NUM_INTS(int version=CURRENT_SESSION_VERSION) {
        return (version < 3) ? 1 : 2;
    }
    static int  SESSION_NUM_FLOATS(int /*version*/=CURRENT_SESSION_VERSION) { return 0; }

public:
    static void  assign_rname3to1(const ResName& rname, char let,
        bool protein);
    static bool is_gap_character(char c) { return !(std::isalpha(c) || c == '?'); }
    static char  nucleic3to1(const ResName& rn);
    static PolymerType  rname_polymer_type(const ResName& rn) {
        if (protein3to1(rn) != 'X') return PT_AMINO;
        if (nucleic3to1(rn) != 'X') return PT_NUCLEIC;
        return PT_NONE;
    }
    static char  protein3to1(const ResName& rn);
    static char  rname3to1(const ResName& rn);

    Sequence(std::string name = "sequence"): _name(name) {}
    Sequence(const Contents& chars, std::string name = "sequence"):
        _contents(chars), _name(name) {}
    Sequence(const std::vector<ResName>& res_names, std::string name = "sequence");  // 3-letter codes
    virtual  ~Sequence() { } 
    template <class InputIterator> void  assign(InputIterator first,
        InputIterator last) { _clear_cache(); _contents.assign(first, last); }
    Contents::reference  at(Contents::size_type n)
        { _clear_cache(); return _contents.at(n); }
    Contents::const_reference  at(Contents::size_type n) const
        { return _contents.at(n); }
    Contents::reference  back() { _clear_cache(); return _contents.back(); }
    Contents::const_reference  back() const { return _contents.back(); }
    Contents::const_iterator  begin() const { return _contents.begin(); }
    // circular implies that the character sequence has been doubled, and that although
    // the res_map (for StructureSeqs) only explicity refers to the front half, it
    // implicitly also applies to the back half
    bool  circular() const { return _circular; }
    void  clear() { _clear_cache(); _contents.clear(); }
    const Contents&  contents() const { return _contents; }
    void  extend(const char* s) { extend(std::string(s)); }
    void  extend(const std::string& s) { _clear_cache(); for (auto c: s) _contents.push_back(c); }
    Contents::const_iterator  end() const { return _contents.end(); }
    Contents::reference  front() { _clear_cache(); return _contents.front(); }
    Contents::const_reference  front() const { return _contents.front(); }
    unsigned int  gapped_to_ungapped(unsigned int index) const;
    Contents::iterator  insert(Contents::const_iterator pos,
        Contents::size_type n, const Contents::value_type& val)
        { _clear_cache(); return _contents.insert(pos, n, val); }
    virtual bool  is_sequence() const { return true; }
    const std::string&  name() const { return _name; }
    Sequence&  operator+=(const Sequence&);
    void  pop_back() { _clear_cache(); _contents.pop_back(); }
    void  pop_front() { _clear_cache(); _contents.erase(_contents.begin()); }
    void  push_back(char c) { _clear_cache(); _contents.push_back(c); }
    void  push_front(char c);
    virtual void  python_destroyed() { delete this; }
    Contents::const_reverse_iterator  rbegin() const
        { return _contents.rbegin(); }
    Contents::const_reverse_iterator  rend() const { return _contents.rend(); }
    std::vector<std::pair<int,int>>  search(const char* pattern, bool case_sensitive = false) const
        { return search(std::string(pattern), case_sensitive); }
    std::vector<std::pair<int,int>>  search(const std::string& pattern, bool case_sensitive = false) const;
    int  session_num_floats(int version=CURRENT_SESSION_VERSION) const { return SESSION_NUM_FLOATS(version); }
    int  session_num_ints(int version=CURRENT_SESSION_VERSION) const { return SESSION_NUM_INTS(version) + _contents.size(); }
    void  session_restore(int, int**, float**);
    void  session_save(int**, float**) const;
    void  set_circular(bool c) { _circular = c; }
    void  set_name(std::string& name);
    void  set_name(const char* name) { std::string n(name); set_name(n); }
    Contents::size_type  size() const { return _contents.size(); }
    void  swap(Contents& x) { _clear_cache(); _contents.swap(x); }
    const Contents&  ungapped() const;
    unsigned int  ungapped_to_gapped(unsigned int index) const;
};

}  // namespace atomstruct

#endif  // atomstruct_Sequence
