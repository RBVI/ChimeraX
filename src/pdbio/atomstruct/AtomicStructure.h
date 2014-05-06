// vim: set expandtab ts=4 sw=4:
#ifndef atomic_AtomicStructure
#define atomic_AtomicStructure

#include <vector>
#include <string>
#include <map>
#include <memory>

// can't use forward declarations for classes that we will
// use unique pointers for, since template expansion checks
// that they have default delete functions
#include "Atom.h"
#include "Bond.h"
#include "CoordSet.h"
#include "Residue.h"
#include "Sequence.h"
#include "imex.h"

class Element;

class ATOMSTRUCT_IMEX AtomicStructure {
public:
    typedef std::vector<std::unique_ptr<Atom>>  Atoms;
    typedef std::vector<std::unique_ptr<Bond>>  Bonds;
    typedef std::vector<std::unique_ptr<CoordSet>>  CoordSets;
    typedef std::vector<std::vector<Residue*>>  Res_Lists;
    typedef std::vector<std::unique_ptr<Residue>>  Residues;
    typedef std::vector<Sequence>  Sequences;
private:
    CoordSet *  _active_coord_set;
    Atoms  _atoms;
    Bonds  _bonds;
    CoordSets  _coord_sets;
    Residues  _residues;
public:
    AtomicStructure();
    const Atoms &    atoms() const { return _atoms; }
    CoordSet *  active_coord_set() const { return _active_coord_set; };
    bool  asterisks_translated;
    std::map<Residue *, char>  best_alt_locs() const;
    const Bonds &    bonds() const { return _bonds; }
    const CoordSets &  coord_sets() const { return _coord_sets; }
    void  delete_bond(Bond *);
    CoordSet *  find_coord_set(int) const;
    Residue *  find_residue(std::string &chain_id, int pos, char insert) const;
    Residue *  find_residue(std::string &chain_id, int pos, char insert,
        std::string &name) const;
    bool  is_traj;
    bool  lower_case_chains;
    void  make_chains(Res_Lists* chain_members = nullptr,
        Sequences* full_sequences = nullptr) const;
    Atom *  new_atom(std::string &name, Element e);
    Bond *  new_bond(Atom *, Atom *);
    CoordSet *  new_coord_set();
    CoordSet *  new_coord_set(int index);
    CoordSet *  new_coord_set(int index, int size);
    Residue *  new_residue(std::string &name, std::string &chain, int pos, char insert,
        Residue *neighbor=NULL, bool after=true);
    std::map<std::string, std::vector<std::string>> pdb_headers;
    int  pdb_version;
    const Residues &  residues() const { return _residues; }
    void  set_active_coord_set(CoordSet *cs);
    void  use_best_alt_locs();
};

#endif  // atomic_AtomicStructure
