# Build the core C++ library

include(FetchContent)

# --------------------------------------------------------------------
# 1) Fetch ExprTk (header-only) from GitHub
# --------------------------------------------------------------------
FetchContent_Declare(
        exprtk
        GIT_REPOSITORY https://github.com/ArashPartow/exprtk.git
        GIT_TAG f46bffcd6966d38a09023fb37ba9335214c9b959           # or pin to a release tag/commit
)
FetchContent_MakeAvailable(exprtk)


add_library(eventide_core
        src/Sampler.cpp
        src/Scenario.cpp
        src/Criterion.cpp
        src/Collector.cpp
        src/RngEngine.cpp
        src/Simulator.cpp
        src/CompiledExpression.cpp
)
target_include_directories(eventide_core
        PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
        $<BUILD_INTERFACE:${exprtk_SOURCE_DIR}>
)
find_package(Threads REQUIRED)
target_link_libraries(eventide_core PUBLIC Threads::Threads)

# Apply our optimized flags to the core
target_compile_options(eventide_core PRIVATE
        $<$<CONFIG:Release>:-Ofast -march=native -flto -funroll-loops -DNDEBUG>
        $<$<CONFIG:RelWithDebInfo>:-O2 -g -pg>
        $<$<CONFIG:Debug>:-O0 -g>
)

# C++ unit tests
enable_testing()
add_subdirectory(tests)
