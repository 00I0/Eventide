include(FetchContent)
FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/refs/heads/main.zip
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

FetchContent_Declare(
        boost_math
        GIT_REPOSITORY https://github.com/boostorg/math.git
        GIT_TAG boost-1.85.0
)
FetchContent_MakeAvailable(boost_math)

file(GLOB TEST_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")
add_executable(rng_engine_test RngEngine_test.cpp)

target_include_directories(rng_engine_test PRIVATE
        ${boost_math_SOURCE_DIR}/include
)

target_link_libraries(rng_engine_test PRIVATE
        eventide_core
        GTest::gtest_main
)


add_executable(benchmark benchmark.cpp)
target_link_libraries(benchmark PRIVATE
        eventide_core
)
