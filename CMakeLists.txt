cmake_minimum_required(VERSION 3.24)

project(inference_server VERSION 0.1.0 LANGUAGES CXX)

option(INFERENCE_BUILD_TESTS "Build the fast, provider-independent unit tests" ON)
option(INFERENCE_BUILD_SERVER "Build the ONNX Runtime-backed inference server" ON)

set(ONNXRUNTIME_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/.deps/onnxruntime" CACHE PATH
    "Path to an ONNX Runtime C/C++ distribution")

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

if(INFERENCE_BUILD_SERVER)
    find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h
        HINTS "${ONNXRUNTIME_ROOT}/include" NO_DEFAULT_PATH)
    find_library(ONNXRUNTIME_LIBRARY onnxruntime
        HINTS "${ONNXRUNTIME_ROOT}/lib" NO_DEFAULT_PATH)
    if(NOT ONNXRUNTIME_INCLUDE_DIR OR NOT ONNXRUNTIME_LIBRARY)
        message(FATAL_ERROR
            "ONNX Runtime was not found. Run scripts/setup_onnxruntime.sh or set ONNXRUNTIME_ROOT.")
    endif()

    add_library(onnxruntime SHARED IMPORTED GLOBAL)
    set_target_properties(onnxruntime PROPERTIES
        IMPORTED_LOCATION "${ONNXRUNTIME_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${ONNXRUNTIME_INCLUDE_DIR}"
    )

    include(FetchContent)

    find_package(nlohmann_json 3.12 CONFIG QUIET)
    if(NOT nlohmann_json_FOUND)
        FetchContent_Declare(
            nlohmann_json
            URL https://github.com/nlohmann/json/releases/download/v3.12.0/json.tar.xz
            DOWNLOAD_EXTRACT_TIMESTAMP TRUE
        )
        FetchContent_MakeAvailable(nlohmann_json)
    endif()

    FetchContent_Declare(
        stb
        URL https://github.com/nothings/stb/archive/013ac3beddff3dbffafd5177e7972067cd2b5083.tar.gz
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )
    FetchContent_MakeAvailable(stb)
    add_library(stb_image INTERFACE)
    target_include_directories(stb_image SYSTEM INTERFACE "${stb_SOURCE_DIR}")

    set(HTTPLIB_USE_OPENSSL_IF_AVAILABLE OFF CACHE BOOL "Disable unused HTTP TLS support" FORCE)
    set(HTTPLIB_USE_ZLIB_IF_AVAILABLE OFF CACHE BOOL "Disable unused HTTP compression" FORCE)
    set(HTTPLIB_USE_BROTLI_IF_AVAILABLE OFF CACHE BOOL "Disable unused HTTP compression" FORCE)
    set(HTTPLIB_USE_ZSTD_IF_AVAILABLE OFF CACHE BOOL "Disable unused HTTP compression" FORCE)
    set(HTTPLIB_INSTALL OFF CACHE BOOL "Disable cpp-httplib install rules" FORCE)
    set(HTTPLIB_TEST OFF CACHE BOOL "Disable upstream cpp-httplib tests" FORCE)
    FetchContent_Declare(
        cpp_httplib
        URL https://github.com/yhirose/cpp-httplib/archive/refs/tags/v0.58.0.tar.gz
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )
    FetchContent_MakeAvailable(cpp_httplib)

    FetchContent_Declare(
        CLI11
        URL https://github.com/CLIUtils/CLI11/archive/refs/tags/v2.6.2.tar.gz
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )
    FetchContent_MakeAvailable(CLI11)

    add_library(inference_runtime
        src/dynamic_batcher.cpp
        src/image_classifier.cpp
        src/metrics.cpp
        src/model_runner.cpp
        src/stb_image_implementation.cpp
    )
    target_link_libraries(inference_runtime PUBLIC
        inference_core
        nlohmann_json::nlohmann_json
        onnxruntime
        stb_image
    )
    set_target_properties(inference_runtime PROPERTIES
        BUILD_RPATH "${ONNXRUNTIME_ROOT}/lib"
    )

    add_library(inference_http src/server.cpp)
    target_link_libraries(inference_http PUBLIC inference_runtime httplib::httplib)

    add_executable(inference-server src/main.cpp)
    target_link_libraries(inference-server PRIVATE inference_http CLI11::CLI11)
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

    if(INFERENCE_BUILD_SERVER)
        add_executable(model_runner_tests tests/model_runner_tests.cpp)
        target_link_libraries(model_runner_tests PRIVATE inference_runtime GTest::gtest_main)
        gtest_discover_tests(model_runner_tests)

        add_executable(image_classifier_tests tests/image_classifier_tests.cpp)
        target_link_libraries(image_classifier_tests PRIVATE inference_runtime GTest::gtest_main)
        gtest_discover_tests(image_classifier_tests)

        add_executable(dynamic_batcher_tests tests/dynamic_batcher_tests.cpp)
        target_link_libraries(dynamic_batcher_tests PRIVATE inference_runtime GTest::gtest_main)
        gtest_discover_tests(dynamic_batcher_tests)

        add_executable(metrics_tests tests/metrics_tests.cpp)
        target_link_libraries(metrics_tests PRIVATE inference_runtime GTest::gtest_main)
        gtest_discover_tests(metrics_tests)

        if(INFERENCE_BUILD_SERVER)
            add_executable(server_tests tests/server_tests.cpp)
            target_link_libraries(server_tests PRIVATE inference_http GTest::gtest_main)
            gtest_discover_tests(server_tests)
        endif()
    endif()
endif()
