// vi: set expandtab ts=4 sw=4:
#ifndef atomstruct_Pseudobond
#define atomstruct_Pseudobond

#include "imex.h"
#include "Connection.h"

// "forward declare" PyObject, which is a typedef of a struct,
// as per the python mailing list:
// http://mail.python.org/pipermail/python-dev/2003-August/037601.html
#ifndef PyObject_HEAD
struct _object;
typedef _object PyObject;
#endif
    
namespace atomstruct {

class Atom;
class ChangeTracker;
class GraphicsContainer;
class Group;

class ATOMSTRUCT_IMEX Pseudobond: public Connection
{
public:
    friend class PBGroup;
    friend class StructurePBGroup;
    friend class CS_PBGroup;

private:
    Group*  _group;

    Pseudobond(Atom* a1, Atom* a2, Group* grp): Connection(a1, a2), _group(grp) {
            _halfbond = false;
            _radius = 0.05;
        };

    // convert a global pb_manager version# to version# for Connection base class
    static int  session_base_version(int /*version*/) { return 1; }
    // version "0" means latest version
    static int  SESSION_NUM_INTS(int /*version*/=0) { return 1; }
    static int  SESSION_NUM_FLOATS(int /*version*/=0) { return 0; }
protected:
    const char*  err_msg_loop() const
        { return "Can't form pseudobond to itself"; }
    const char*  err_msg_not_end() const
        { return "Atom given to other_end() not in pseudobond!"; }
public:
    virtual ~Pseudobond() {}
    ChangeTracker*  change_tracker() const;
    GraphicsContainer*  graphics_container() const;
    Group*  group() const { return _group; }
    // version "0" means latest version
    static int  session_num_floats(int version=0) {
        return SESSION_NUM_FLOATS(version) + Connection::session_num_floats(version);
    }
    static int  session_num_ints(int version=0) {
        return SESSION_NUM_INTS(version) + Connection::session_num_ints(version);
    }
    void  session_restore(int version, int** ints, float** floats);
    void  session_save(int** ints, float** floats) const;
    void  track_change(const std::string& reason) const {
        change_tracker()->add_modified(this, reason);
    }
};

}  // namespace atomstruct

#endif  // atomstruct_Pseudobond
