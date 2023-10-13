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

#ifndef atomstruct_Rgba
#define atomstruct_Rgba

#include <initializer_list>

namespace atomstruct {
    
class ATOMSTRUCT_IMEX Rgba {
public:
    typedef unsigned char  Channel;
    Channel  r, g, b, a;
    Rgba(): r(255), g(255), b(255), a(255) {}
    Rgba(Channel red, Channel green, Channel blue, Channel alpha = 255) {
        r = red; g = green; b = blue; a = alpha;
    }
    Rgba(std::initializer_list<Channel> const rgba) {
#ifdef Cpp14
        static_assert(rgba.size() != 4,
            "Rgba initializer list must have exactly 4 values");
#endif
        auto i = rgba.begin();
        r = *i++;
        g = *i++;
        b = *i++;
        a = *i;
    }
    bool  operator==(const Rgba& other) const {
        return other.r == r && other.g == g && other.b == b && other.a == a;
    }

    static const int  SESSION_NUM_INTS = 4;
    static const int  SESSION_NUM_FLOATS = 0;
    static int  session_num_floats() { return SESSION_NUM_FLOATS; }
    static int  session_num_ints() { return SESSION_NUM_INTS; }
    void  session_restore(int** ints, float** ) {
        auto& int_ptr = *ints;
        r = int_ptr[0]; g = int_ptr[1]; b = int_ptr[2]; a = int_ptr[3];
        int_ptr += SESSION_NUM_INTS;
    }
    void  session_save(int** ints, float** ) const {
        auto& int_ptr = *ints;
        int_ptr[0] = r; int_ptr[1] = g; int_ptr[2] = b; int_ptr[3] = a;
        int_ptr += SESSION_NUM_INTS;
    }
};

} //  namespace atomstruct

#endif  // atomstruct_Rgba
