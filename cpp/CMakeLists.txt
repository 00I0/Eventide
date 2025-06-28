# Build the core C++ library
add_library(eventide_core
        src/Parameter.cpp
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
