// Compile contourdata for the _flying_edges shared object.
// The table data lives in _map/contourdata.cpp; we include it here
// so that _flying_edges links its own copy without cross-module dependencies.
#include "../_map/contourdata.cpp"
