cmake_minimum_required(VERSION 3.20)
project(OpenShapeMatch VERSION 0.3.0 LANGUAGES CXX)

include(CheckCXXCompilerFlag)

option(OPENSHAPE_ENABLE_AVX2 "Compile the AVX2 score kernel when supported" ON)
option(OPENSHAPE_USE_AVX2_BY_DEFAULT "Select AVX2 at runtime when available" OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT OpenCV_DIR AND DEFINED ENV{HOME})
  set(_local_opencv_dir "$ENV{HOME}/.local/opencv-4.10.0/lib/cmake/opencv4")
  if(EXISTS "${_local_opencv_dir}/OpenCVConfig.cmake")
    set(OpenCV_DIR "${_local_opencv_dir}" CACHE PATH "Local OpenCV CMake directory")
    message(STATUS "Using user-local OpenCV: ${OpenCV_DIR}")
  endif()
endif()

find_package(OpenCV 4 QUIET COMPONENTS core imgproc imgcodecs)
if(NOT OpenCV_FOUND)
  message(FATAL_ERROR "OpenCV 4 development package (core, imgproc, imgcodecs) is required. Set OpenCV_DIR or install libopencv-dev.")
endif()
find_package(Threads REQUIRED)

add_library(openshape_match
    src/core/image.cpp
    src/core/cpu_features.cpp
    src/edge/edge_engine.cpp
    src/model/shape_model.cpp
    src/pyramid/pyramid.cpp
    src/matcher/matcher.cpp
    src/matcher/exhaustive_matcher.cpp
)
add_library(openshape::OpenShapeMatch ALIAS openshape_match)
target_include_directories(openshape_match PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(openshape_match PUBLIC ${OpenCV_LIBS} Threads::Threads)
target_compile_features(openshape_match PUBLIC cxx_std_17)
set_target_properties(openshape_match PROPERTIES
    EXPORT_NAME OpenShapeMatch
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
    WINDOWS_EXPORT_ALL_SYMBOLS ON
    POSITION_INDEPENDENT_CODE ON
)
if(MSVC)
  target_compile_options(openshape_match PRIVATE /W4 /permissive-)
else()
  target_compile_options(openshape_match PRIVATE -Wall -Wextra -Wpedantic)
endif()

# Compile ISA-specific code in a separate object target. The main library
# remains portable and only calls this object after runtime feature probing.
if(OPENSHAPE_ENABLE_AVX2 AND NOT MSVC)
  check_cxx_compiler_flag("-mavx2" OPENSHAPE_COMPILER_HAS_MAVX2)
  if(OPENSHAPE_COMPILER_HAS_MAVX2)
    add_library(openshape_avx2 OBJECT src/matcher/score_avx2.cpp)
    target_include_directories(openshape_avx2 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include ${OpenCV_INCLUDE_DIRS})
    target_compile_features(openshape_avx2 PRIVATE cxx_std_17)
    target_compile_options(openshape_avx2 PRIVATE -mavx2)
    target_sources(openshape_match PRIVATE $<TARGET_OBJECTS:openshape_avx2>)
    target_compile_definitions(openshape_match PRIVATE OPENSHAPE_HAS_AVX2_KERNEL=1)
    if(OPENSHAPE_USE_AVX2_BY_DEFAULT)
      target_compile_definitions(openshape_match PRIVATE OPENSHAPE_USE_AVX2_BY_DEFAULT=1)
    endif()
  endif()
endif()

add_executable(shape_match_example examples/basic_match.cpp)
target_link_libraries(shape_match_example PRIVATE openshape_match)

option(BUILD_BENCHMARK "Build the lightweight matching benchmark" OFF)
if(BUILD_BENCHMARK)
  add_executable(shape_match_benchmark benchmark/benchmark.cpp)
  target_link_libraries(shape_match_benchmark PRIVATE openshape_match)
endif()

include(CTest)
if(BUILD_TESTING)
  add_executable(openshape_smoke tests/smoke.cpp)
  target_link_libraries(openshape_smoke PRIVATE openshape_match)
  add_test(NAME openshape_smoke COMMAND openshape_smoke)
  add_executable(openshape_unit_tests tests/unit_tests.cpp)
  target_link_libraries(openshape_unit_tests PRIVATE openshape_match)
  add_test(NAME openshape_unit_tests COMMAND openshape_unit_tests)
  add_executable(openshape_integration_tests tests/integration_tests.cpp)
  target_link_libraries(openshape_integration_tests PRIVATE openshape_match)
  add_test(NAME openshape_integration_tests COMMAND openshape_integration_tests)
  add_test(NAME openshape_package_consumer
    COMMAND ${CMAKE_COMMAND}
      -DOPENSHAPE_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}
      -DOPENSHAPE_BINARY_DIR=${CMAKE_CURRENT_BINARY_DIR}
      -DOPENSHAPE_OPENCV_DIR=${OpenCV_DIR}
      -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/package_consumer_test.cmake)
  find_package(GTest QUIET)
  if(GTest_FOUND)
    file(GLOB TEST_SOURCES CONFIGURE_DEPENDS tests/unit/*.cpp tests/integration/*.cpp)
    if(TEST_SOURCES)
      add_executable(openshape_tests ${TEST_SOURCES})
      target_link_libraries(openshape_tests PRIVATE openshape_match GTest::gtest_main)
      include(GoogleTest)
      gtest_discover_tests(openshape_tests)
    endif()
  else()
    message(STATUS "GTest not found; no C++ tests will be built")
  endif()
endif()

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
install(TARGETS openshape_match EXPORT OpenShapeMatchTargets
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(EXPORT OpenShapeMatchTargets NAMESPACE openshape:: DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/OpenShapeMatch)
configure_package_config_file(cmake/OpenShapeMatchConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/OpenShapeMatchConfig.cmake
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/OpenShapeMatch)
write_basic_package_version_file(${CMAKE_CURRENT_BINARY_DIR}/OpenShapeMatchConfigVersion.cmake
  VERSION ${PROJECT_VERSION} COMPATIBILITY SameMajorVersion)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/OpenShapeMatchConfig.cmake
              ${CMAKE_CURRENT_BINARY_DIR}/OpenShapeMatchConfigVersion.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/OpenShapeMatch)
