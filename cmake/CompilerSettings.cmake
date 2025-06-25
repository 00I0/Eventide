set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)


if (NOT CMAKE_CONFIGURATION_TYPES)  # single-config
    set(CMAKE_CXX_FLAGS_RELEASE "-Ofast -march=native -flto -funroll-loops -DNDEBUG")
    set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g -pg")
    set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g")
else ()                            # multi-config
    foreach (cfg IN LISTS CMAKE_CONFIGURATION_TYPES)
        string(TOUPPER "${cfg}" ucfg)
        set(CMAKE_CXX_FLAGS_${ucfg} "-O2 -g")
    endforeach ()

endif ()