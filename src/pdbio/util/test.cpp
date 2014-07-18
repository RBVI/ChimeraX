#include <iostream>
#include <vector>
#include <ctime>
#include <stdio.h>

#define QUARTER 2500000
#define RESERVE_SIZE 4

#define USE_VECTOR_MAP

#ifdef USE_VECTOR_MAP
#include "VectorMap.h"
typedef util::VectorMap<int, void*>  MapType;
#endif

#ifdef USE_MAP
#include <map>
typedef std::map<int, void*>  MapType;
#endif

#ifdef USE_UNORDERED_MAP
#include <unordered_map>
typedef std::unordered_map<int, void*>  MapType;
#endif

int main()
{
	std::cout << "mem check 1: "; getchar();
	clock_t start = clock();
	//std::vector<MapType*> items;
	for (int i=0; i<QUARTER; ++i) {
		MapType* item = new MapType;
		item->reserve(RESERVE_SIZE);
		//items.push_back(item);
	}
	for (int i=0; i<QUARTER; ++i) {
		MapType* item = new MapType;
		item->reserve(RESERVE_SIZE);
		(*item)[0] = nullptr;
		//items.push_back(item);
	}
	for (int i=0; i<QUARTER; ++i) {
		MapType* item = new MapType;
		item->reserve(RESERVE_SIZE);
		(*item)[0] = nullptr;
		(*item)[1] = nullptr;
		//items.push_back(item);
	}
	for (int i=0; i<QUARTER; ++i) {
		MapType* item = new MapType;
		item->reserve(RESERVE_SIZE);
		(*item)[0] = nullptr;
		(*item)[1] = nullptr;
		(*item)[2] = nullptr;
		//items.push_back(item);
	}
	std::cout << (clock() - start) / (float)CLOCKS_PER_SEC << "\n";
	std::cout << "mem check 2: "; getchar();
	return 0;
}
