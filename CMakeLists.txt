cmake_minimum_required(VERSION 3.15)
project(Eventide LANGUAGES CXX)

if (POLICY CMP0148)
    cmake_policy(SET CMP0148 NEW)
endif ()

# 2) Find the Python from the venv
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
message(STATUS "Using Python3: ${Python3_EXECUTABLE}")


# Load our custom flags
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
include(CompilerSettings)

# Pull in the C++ core
add_subdirectory(cpp)

# Pull in the Python bindings
add_subdirectory(bindings)

add_custom_target(python_install ALL
        COMMAND ${Python3_EXECUTABLE} -m pip install -e ${CMAKE_SOURCE_DIR}/python/eventide
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/python/eventide
        DEPENDS _eventide
)
