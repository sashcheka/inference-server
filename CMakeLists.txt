cmake_minimum_required(VERSION 3.24)

project(inference_server VERSION 0.1.0 LANGUAGES CXX)

option(INFERENCE_BUILD_TESTS "Build the fast, provider-independent unit tests" ON)

add_library(inference_core INTERFACE)
target_compile_features(inference_core INTERFACE cxx_std_20)
target_include_directories(inference_core INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(inference_core INTERFACE
        -Wall
        -Wextra
        -Wpedantic
        -Wconversion
        -Wsign-conversion
    )
endif()

if(INFERENCE_BUILD_TESTS)
    include(CTest)
    include(FetchContent)

    find_package(GTest CONFIG QUIET)
    if(NOT GTest_FOUND)
        set(INSTALL_GTEST OFF CACHE BOOL "Disable GoogleTest installation" FORCE)
        set(gtest_force_shared_crt ON CACHE BOOL "Use shared CRT on Windows" FORCE)
        FetchContent_Declare(
            googletest
            GIT_REPOSITORY https://github.com/google/googletest.git
            GIT_TAG v1.17.0
            GIT_SHALLOW TRUE
        )
        FetchContent_MakeAvailable(googletest)
    endif()

    add_executable(bounded_queue_tests tests/bounded_queue_tests.cpp)
    target_link_libraries(bounded_queue_tests PRIVATE inference_core GTest::gtest_main)
    include(GoogleTest)
    gtest_discover_tests(bounded_queue_tests)
endif()
