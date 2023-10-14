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

#ifndef atomstruct_Connection
#define atomstruct_Connection

#include <stdexcept>

#include "ChangeTracker.h"
#include "Coord.h"
#include "destruct.h"
#include "Real.h"
#include "Rgba.h"
#include "session.h"

// "forward declare" PyObject, which is a typedef of a struct,
// as per the python mailing list:
// http://mail.python.org/pipermail/python-dev/2003-August/037601.html
#ifndef PyObject_HEAD
struct _object;
typedef _object PyObject;
#endif
    
namespace atomstruct {

class Atom;
class GraphicsChanges;

class ATOMSTRUCT_IMEX Connection {
public:
    typedef Atom*  Atoms[2];

    static int  SESSION_NUM_INTS(int version=CURRENT_SESSION_VERSION) {
        return version < 11 ? 3 : 4;
    }
    static int  SESSION_NUM_FLOATS(int /*version*/=CURRENT_SESSION_VERSION) { return 1; }
protected:
    virtual const char*  err_msg_loop() const
        { return "Can't connect atom to itself"; }
    virtual const char*  err_msg_not_in_connection() const
        { return "Atom arg of other_atom() not in bond/pseudobond"; }

    Atoms  _atoms;

    bool  _display = true;
    int  _hide = 0;
    bool  _halfbond = true;
    float  _radius = 0.2f;
    Rgba  _rgba;
    bool  _selected = false;
public:
    Connection(Atom* a1, Atom* a2) { _atoms[0] = a1; _atoms[1] = a2; }

    virtual void  finish_construction(); // virtual calls now working...
    virtual  ~Connection() { auto du = DestructionUser(this); }
    bool  contains(Atom* a) const { return a == _atoms[0] || a == _atoms[1]; }
    const Atoms&  atoms() const { return _atoms; }
    Real  length() const;
    Atom*  other_atom(const Atom* a) const;
    void  session_info(bool intra_mol, PyObject* ints, PyObject* floats, PyObject* misc) const;
    Real  sqlength() const;

    // session related
    static int  session_num_floats(int version=CURRENT_SESSION_VERSION) {
        return SESSION_NUM_FLOATS(version) + Rgba::session_num_floats();
    }
    static int  session_num_ints(int version=CURRENT_SESSION_VERSION) {
        return SESSION_NUM_INTS(version) + Rgba::session_num_ints();
    }
    void  session_restore(int version, int** ints, float** floats) {
        _rgba.session_restore(ints, floats);
        auto& int_ptr = *ints;
        _display = int_ptr[0];
        _hide = int_ptr[1];
        _halfbond = int_ptr[2];
        _selected = version < 11 ? false : int_ptr[3];
        int_ptr += SESSION_NUM_INTS(version);

        auto& float_ptr = *floats;
        _radius = float_ptr[0];
        float_ptr += SESSION_NUM_FLOATS(version);
    }
    void  session_save(int** ints, float** floats) const {
        _rgba.session_save(ints, floats);
        auto& int_ptr = *ints;
        int_ptr[0] = _display;
        int_ptr[1] = _hide;
        int_ptr[2] = _halfbond;
        int_ptr[3] = _selected;
        int_ptr += SESSION_NUM_INTS();

        auto& float_ptr = *floats;
        float_ptr[0] = _radius;
        float_ptr += SESSION_NUM_FLOATS();
    }

    // change tracking
    virtual void  track_change(const std::string& reason) const = 0;

    // graphics related
    const Rgba&  color() const { return _rgba; }
    bool  display() const { return _display; }
    bool  halfbond() const { return _halfbond; }
    int  hide() const { return _hide; }
    virtual GraphicsChanges*  graphics_changes() const = 0;
    float  radius() const { return _radius; }
    void  set_color(Rgba::Channel r, Rgba::Channel g, Rgba::Channel b, Rgba::Channel a)
        { set_color(Rgba({r, g, b, a})); }
    void  set_color(const Rgba& rgba);
    void  set_display(bool d);
    void  set_halfbond(bool hb);
    void  set_hide(int h);
    void  set_hide_bits(int bit_mask) { set_hide(hide() | bit_mask); }
    void  clear_hide_bits(int bit_mask) { set_hide(hide() & ~bit_mask); }
    void  set_radius(float r);
    virtual bool shown() const;
    bool  visible() const { return _hide ? false : _display; }
    bool  selected() const { return _selected; }
    void  set_selected(bool s);
};

class ATOMSTRUCT_IMEX UniqueConnection: public Connection {
protected:
    virtual const char*  err_msg_exists() const
        { return "Connection already exists between atoms"; }
public:
    UniqueConnection(Atom *a1, Atom *a2) : Connection(a1, a2) {}
    virtual void  add_to_atoms() = 0;
    void  finish_construction(); // virtual calls now working...
    virtual  ~UniqueConnection() {}
};

inline Atom *
Connection::other_atom(const Atom *a) const
{
    if (a == _atoms[0])
        return _atoms[1];
    if (a == _atoms[1])
        return _atoms[0];
    throw std::invalid_argument(err_msg_not_in_connection());
}

} //  namespace atomstruct

#include "Atom.h"
#include "Structure.h"

namespace atomstruct {

inline void
Connection::finish_construction()
{
    if (_atoms[0] == _atoms[1])
        throw std::invalid_argument(err_msg_loop());
    graphics_changes()->set_gc_shape();
}

inline Real
Connection::length() const {
    return _atoms[0]->coord().distance(_atoms[1]->coord());
}

inline void
Connection::set_display(bool d) {
    if (d == _display)
        return;
    graphics_changes()->set_gc_shape();
    graphics_changes()->set_gc_display();
    track_change(ChangeTracker::REASON_DISPLAY);
    _display = d;
}

inline void
Connection::set_color(const Rgba& rgba) {
    if (rgba == _rgba)
        return;
    graphics_changes()->set_gc_color();
    track_change(ChangeTracker::REASON_COLOR);
    _rgba = rgba;
}

inline void
Connection::set_halfbond(bool hb) {
    if (hb == _halfbond)
        return;
    graphics_changes()->set_gc_color();
    track_change(ChangeTracker::REASON_HALFBOND);
    _halfbond = hb;
}

inline void
Connection::set_hide(int h) {
    if (h == _hide)
        return;
    graphics_changes()->set_gc_shape();
    track_change(ChangeTracker::REASON_HIDE);
    _hide = h;
}

inline void
Connection::set_radius(float r) {
    if (r == _radius)
        return;
    graphics_changes()->set_gc_shape();
    track_change(ChangeTracker::REASON_RADIUS);
    _radius = r;
}

inline bool
Connection::shown() const {
    return visible() && _atoms[0]->visible() && _atoms[1]->visible();
}

inline Real
Connection::sqlength() const {
    return _atoms[0]->coord().sqdistance(_atoms[1]->coord());
}

inline void
UniqueConnection::finish_construction()
{
    Connection::finish_construction();
    Atom* a1 = _atoms[0];
    Atom* a2 = _atoms[1];
    if (a1->connects_to(a2))
        throw std::invalid_argument(err_msg_exists());
    add_to_atoms();
}

inline void
Connection::set_selected(bool s)
{
    if (s == _selected)
        return;
    graphics_changes()->set_gc_select();
    track_change(ChangeTracker::REASON_SELECTED);
    _selected = s;
}

} //  namespace atomstruct

#endif  // atomstruct_Connection
