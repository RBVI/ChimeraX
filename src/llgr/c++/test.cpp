#include <iostream>
#include <typeinfo>

enum DataType {
	Float, Vec2, Vec3, Vec4,
	Int, IVec2, IVec3, IVec4,
	Bool, BVec2, BVec3, BVec4,
	Mat2, Mat3, Mat4,
#ifdef HAVE_TEXTURE
	Sampler1D, Sampler2D, Sampler3D, SamplerCube, 
	Sampler1DShadow, Sampler2DShadow,
#endif
	Unknown
};

template <typename T>
struct TypeToDataType;

template <>
struct TypeToDataType<float *> { static const DataType value = Float; };
template <>
struct TypeToDataType<float [2]> { static const DataType value = Vec2; };
template <>
struct TypeToDataType<float [3]> { static const DataType value = Vec3; };
template <>
struct TypeToDataType<float [4]> { static const DataType value = Vec4; };
template <>
struct TypeToDataType<float [4][4]> { static const DataType value = Mat4; };

int
save(DataType t, const void *data)
{
	return t;
}

template <typename T>
int save(const T &data)
{
	return save(TypeToDataType<T>::value, data);
}

int
main()
{
	float f[3] = { 0, 1, 3.5 };
	float m[4][4] = { 1 };
	std::cout << "should output 2 14\n";
	std::cout << save(f) << ' ' << save(m) << '\n';
}
