// vi: set expandtab ts=4 sw=4:
#ifndef basegeom_Rgba
#define basegeom_Rgba

namespace basegeom {
    
class Rgba {
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
    void  session_save(int** ints, float** ) const {
        auto& int_ptr = *ints;
        int_ptr[0] = r; int_ptr[1] = g; int_ptr[2] = b; int_ptr[3] = a;
        int_ptr += SESSION_NUM_INTS;
    }
};

} //  namespace basegeom

#endif  // basegeom_Rgba
