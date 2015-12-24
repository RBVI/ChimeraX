// vi: set expandtab ts=4 sw=4:
#ifndef atomstruct_Pseudobond
#define atomstruct_Pseudobond

#include <basegeom/Connection.h>
#include "imex.h"

// "forward declare" PyObject, which is a typedef of a struct,
// as per the python mailing list:
// http://mail.python.org/pipermail/python-dev/2003-August/037601.html
#ifndef PyObject_HEAD
struct _object;
typedef _object PyObject;
#endif
    
namespace atomstruct {

class Atom;
class Group;

using basegeom::ChangeTracker;
using basegeom::GraphicsContainer;

class ATOMSTRUCT_IMEX Pseudobond: public basegeom::Connection<Atom>
{
public:
    friend class PBGroup;
    friend class StructurePBGroup;
    friend class CS_PBGroup;

    static const int  SESSION_NUM_INTS = 0;
    static const int  SESSION_NUM_FLOATS = 0;
private:
    Group*  _group;

    Pseudobond(Atom* a1, Atom* a2, Group* grp):
        basegeom::Connection<Atom>(a1, a2), _group(grp) {
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
    GraphicsContainer*  graphics_container() const;
    Group*  group() const { return _group; }
    static int  session_num_floats() {
        return SESSION_NUM_FLOATS + Connection<Atom>::session_num_floats();
    }
    static int  session_num_ints() {
        return SESSION_NUM_INTS + Connection<Atom>::session_num_ints();
    }
    void  session_restore(int** ints, float** floats) {
        basegeom::Connection<Atom>::session_restore(ints, floats);
    }
    void  session_save(int** ints, float** floats) const {
        basegeom::Connection<Atom>::session_save(ints, floats);
    }
    void  track_change(const std::string& reason) const {
        change_tracker()->add_modified(this, reason);
    }
};

}  // namespace atomstruct

#endif  // atomstruct_Pseudobond
