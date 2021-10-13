include(${CMAKE_CURRENT_SOURCE_DIR}/../CMakeLists.bundle)
cmake_minimum_required(VERSION 3.10)
project(BlastProtein VERSION 2.0)

string(TOLOWER ${CMAKE_PROJECT_NAME} PKG_NAME)

add_custom_command(
	PRE_BUILD
	OUTPUT bundle_info.xml
	DEPENDS bundle_info.xml.in
	COMMAND sed -e 's/VERSION/${CMAKE_PROJECT_VERSION}/' -e 's/BUNDLE_NAME/${CMAKE_PROJECT_NAME}/' -e 's/PKG_NAME/${PKG_NAME}/' bundle_info.xml.in > bundle_info.xml
)
