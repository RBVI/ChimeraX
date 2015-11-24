// vi: set expandtab ts=4 sw=4:
#ifndef atomstruct_Pseudobond
#define atomstruct_Pseudobond

#include <basegeom/Connection.h>
#include "imex.h"

namespace atomstruct {

class Atom;

using basegeom::ChangeTracker;
using basegeom::GraphicsContainer;

class ATOMSTRUCT_IMEX Pseudobond: public basegeom::Connection<Atom>
{
    friend class PBGroup;
    friend class StructurePBGroup;
    friend class CS_PBGroup;
private:
    GraphicsContainer*  _gc;

    Pseudobond(Atom* a1, Atom* a2, GraphicsContainer* gc):
        basegeom::Connection<Atom>(a1, a2), _gc(gc) {
            _halfbond = false;
            _radius = 0.05;
        };
protected:
    const char*  err_msg_loop() const
        { return "Can't form pseudobond to itself"; }
    const char*  err_msg_not_end() const
        { return "Atom given to other_end() not in pseudobond!"; }
public:
    virtual ~Pseudobond() {}
    typedef End_points  Atoms;
    const Atoms&  atoms() const { return end_points(); }
    ChangeTracker*  change_tracker() const;
    GraphicsContainer*  graphics_container() const { return _gc; }
    GraphicsContainer*  group() const { return graphics_container(); }
    void track_change(const std::string& reason) const {
        change_tracker()->add_modified(this, reason);
    }
};

}  // namespace atomstruct

#endif  // atomstruct_Pseudobond
