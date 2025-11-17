cmake_minimum_required (VERSION 3.0)
project(malfeasance)

set(TOP_LEVEL ${CMAKE_CURRENT_SOURCE_DIR})

set(MALFEASANCE_HEADERS fem_grid_pml.h)
set(MALFEASANCE_SOURCES fem_grid_pml.c)

add_library(malfeasance SHARED ${MALFEASANCE_SOURCES} ${MALFEASANCE_HEADERS})

add_executable(malfeasance_frontend malfeasance_frontend.cpp ${MALFEASANCE_SOURCES} ${MALFEASANCE_HEADERS})
set_property(TARGET malfeasance_frontend PROPERTY CXX_STANDARD 17)
if (CMAKE_COMPILER_IS_GNUCC)
    if (NOT CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 8.0)
        target_link_libraries(malfeasance_frontend PRIVATE stdc++fs)
    endif()
endif()
target_link_libraries(malfeasance_frontend PRIVATE pthread)

set_property(TARGET malfeasance
             APPEND
             PROPERTY INTERFACE_INCLUDE_DIRECTORIES
                      "${TOP_LEVEL}")

set_property(TARGET malfeasance
             APPEND
	     PROPERTY INTERFACE_LINK_LIBRARIES pthread) 

if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Ofast -fvectorize -Rpass-missed=loop-vectorize")
    set(CMAKE_CFLAGS_RELEASE "${CMAKE_CFLAGS_RELEASE} -Ofast -fvectorize -Rpass-missed=loop-vectorize")
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Ofast -ftree-vectorize")
    set(CMAKE_CFLAGS_RELEASE "${CMAKE_CFLAGS_RELEASE} -Ofast -ftree-vectorize")
endif()

