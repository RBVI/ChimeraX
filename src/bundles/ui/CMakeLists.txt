include(${CMAKE_CURRENT_SOURCE_DIR}/../CMakeLists.bundle)
cmake_minimum_required(VERSION 3.10)
project(UI VERSION 1.13.4)

string(TOLOWER ${CMAKE_PROJECT_NAME} PKG_NAME)

set(CLEAN_ALWAYS src/about.html file(GLOB src/*/__pycache__))
set(DATAFILES (file(GLOB src/icons/*.png) file(GLOB src/mousemode_icons/*.png) file(GLOB src/widgets/icons/*.png)))

add_custom_command(
	PRE_BUILD
	OUTPUT bundle_info.xml
	DEPENDS bundle_info.xml.in
	COMMAND sed -e 's/VERSION/${CMAKE_PROJECT_VERSION}/' -e 's/BUNDLE_NAME/${CMAKE_PROJECT_NAME}/' -e 's/PKG_NAME/${PKG_NAME}/' bundle_info.xml.in > bundle_info.xml
	COMMENT "Generating bundle_info.xml"
)

add_custom_command(
	OUTPUT src/about.html
	DEPENDS src/about.html.in 
	COMMAND sed -e '/===/d' -e 's,https:.*html,<a href="&">&</a>.,' ${TOP}/copyright.txt | sed -e '/COPYRIGHT_NOTICE/r /dev/stdin' -e '/COPYRIGHT_NOTICE/d' src/about.html.in > src/about.html 
	COMMENT "Generating about.html"
)

add_custom_target(generate_about DEPENDS src/about.html)
add_dependencies(app-install generate_about)
add_dependencies(wheel generate_about)


