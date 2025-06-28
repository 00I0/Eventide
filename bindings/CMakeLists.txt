cmake_minimum_required(VERSION 3.15)

include(FetchContent)
FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG v2.11.1
)

FetchContent_MakeAvailable(pybind11)


pybind11_add_module(_eventide binding.cpp)
target_link_libraries(_eventide PRIVATE eventide_core)


set_target_properties(_eventide PROPERTIES
        OUTPUT_NAME "_eventide"                              # module name
        PREFIX ""                                            # no "lib" prefix
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/python/eventide
)

