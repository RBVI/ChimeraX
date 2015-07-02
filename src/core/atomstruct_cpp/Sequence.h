// vi: set expandtab ts=4 sw=4:
#ifndef atomstruct_Sequence
#define atomstruct_Sequence

#include <vector>
#include <map>
#include <string>

#include "imex.h"
#include "string_types.h"

namespace atomstruct {

class ATOMSTRUCT_IMEX Sequence {
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
    void  _clear_cache() const
        { _cache_ungapped.clear();  _cache_g2ug.clear(); _cache_ug2g.clear(); }
    // can't inherit from vector, since we need to clear caches on changes
    Contents  _contents;
public:
    static void  assign_rname3to1(const ResName& rname, char let,
        bool protein);
    static char  nucleic3to1(const ResName& rn);
    static char  protein3to1(const ResName& rn);
    static char  rname3to1(const ResName& rn);

    Sequence() {}
    Sequence(const std::vector<ResName>& res_names);  // 3-letter codes
    virtual  ~Sequence() {}

    template <class InputIterator> void  assign(InputIterator first,
        InputIterator last) { _clear_cache(); _contents.assign(first, last); }
    Contents::reference  at(Contents::size_type n)
        { _clear_cache(); return _contents.at(n); }
    Contents::const_reference  at(Contents::size_type n) const
        { return _contents.at(n); }
    Contents::reference  back() { _clear_cache(); return _contents.back(); }
    Contents::const_reference  back() const { return _contents.back(); }
    Contents::const_iterator  begin() const { return _contents.begin(); }
    void  clear() { _clear_cache(); _contents.clear(); }
    const Contents&  contents() const { return _contents; }
    Contents::reference  front() { _clear_cache(); return _contents.front(); }
    Contents::const_reference  front() const { return _contents.front(); }
    Contents::const_iterator  end() const { return _contents.end(); }
    unsigned int  gapped_to_ungapped(unsigned int index) const;
    Contents::iterator  insert(Contents::const_iterator pos,
        Contents::size_type n, Contents::value_type val)
        { _clear_cache(); return _contents.insert(pos, n, val); }
    void  pop_back() { _clear_cache(); _contents.pop_back(); }
    void  pop_front() { _clear_cache(); _contents.erase(_contents.begin()); }
    void  push_back(unsigned char c) { _clear_cache(); _contents.push_back(c); }
    Contents::const_reverse_iterator  rbegin() const
        { return _contents.rbegin(); }
    Contents::const_reverse_iterator  rend() const { return _contents.rend(); }
    Contents::size_type  size() const { return _contents.size(); }
    void  swap(Contents& x) { _clear_cache(); _contents.swap(x); }
    const Contents&  ungapped() const;
    unsigned int  ungapped_to_gapped(unsigned int index) const;
};

}  // namespace atomstruct

#endif  // atomstruct_Sequence
