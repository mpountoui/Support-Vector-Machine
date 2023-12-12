cmake_minimum_required(VERSION 3.1)
project(SUPPORT-VECTOR-MACHINE)

include_directories("${CMAKE_SOURCE_DIR}/include")

file (GLOB SOURCE_FILES "src/*.cpp")
file (GLOB HEADER_FILES "include/*.hpp")

# Set up such that XCode organizes the files
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${SOURCE_FILES} ${HEADER_FILES} )

find_package(pybind11 REQUIRED)
pybind11_add_module(mySVM 
	${SOURCE_FILES}
	${HEADER_FILES}
)

target_link_libraries(mySVM PUBLIC)

install(TARGETS mySVM
  	COMPONENT python
  	LIBRARY DESTINATION ${PYTHON_LIBRARY_DIR}
  )