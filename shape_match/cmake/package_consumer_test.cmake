if(NOT DEFINED OPENSHAPE_SOURCE_DIR OR NOT DEFINED OPENSHAPE_BINARY_DIR)
  message(FATAL_ERROR "OPENSHAPE_SOURCE_DIR and OPENSHAPE_BINARY_DIR are required")
endif()

set(prefix "${OPENSHAPE_BINARY_DIR}/package-consumer-prefix")
set(consumer_build "${OPENSHAPE_BINARY_DIR}/package-consumer-build")
file(REMOVE_RECURSE "${prefix}" "${consumer_build}")

execute_process(
  COMMAND "${CMAKE_COMMAND}" --install "${OPENSHAPE_BINARY_DIR}" --prefix "${prefix}"
  RESULT_VARIABLE install_result)
if(NOT install_result EQUAL 0)
  message(FATAL_ERROR "OpenShapeMatch installation failed: ${install_result}")
endif()

set(configure_command "${CMAKE_COMMAND}"
  -S "${OPENSHAPE_SOURCE_DIR}/tests/package_consumer"
  -B "${consumer_build}"
  "-DCMAKE_PREFIX_PATH=${prefix}")
if(DEFINED OPENSHAPE_OPENCV_DIR AND NOT OPENSHAPE_OPENCV_DIR STREQUAL "")
  list(APPEND configure_command "-DOpenCV_DIR=${OPENSHAPE_OPENCV_DIR}")
endif()
execute_process(COMMAND ${configure_command} RESULT_VARIABLE configure_result)
if(NOT configure_result EQUAL 0)
  message(FATAL_ERROR "Package consumer configure failed: ${configure_result}")
endif()

execute_process(
  COMMAND "${CMAKE_COMMAND}" --build "${consumer_build}" --config Release
  RESULT_VARIABLE build_result)
if(NOT build_result EQUAL 0)
  message(FATAL_ERROR "Package consumer build failed: ${build_result}")
endif()

set(consumer_executable "${consumer_build}/openshape_package_consumer")
if(WIN32)
  set(consumer_executable "${consumer_build}/Release/openshape_package_consumer.exe")
endif()
set(run_command "${consumer_executable}")
if(DEFINED OPENSHAPE_OPENCV_DIR AND NOT OPENSHAPE_OPENCV_DIR STREQUAL "")
  get_filename_component(opencv_library_dir "${OPENSHAPE_OPENCV_DIR}/../.." ABSOLUTE)
  if(WIN32)
    set(run_command "${CMAKE_COMMAND}" -E env
      "PATH=${prefix}/bin;${opencv_library_dir};$ENV{PATH}" "${consumer_executable}")
  elseif(APPLE)
    set(run_command "${CMAKE_COMMAND}" -E env
      "DYLD_LIBRARY_PATH=${prefix}/lib:${opencv_library_dir}:$ENV{DYLD_LIBRARY_PATH}"
      "${consumer_executable}")
  else()
    set(run_command "${CMAKE_COMMAND}" -E env
      "LD_LIBRARY_PATH=${prefix}/lib:${opencv_library_dir}:$ENV{LD_LIBRARY_PATH}"
      "${consumer_executable}")
  endif()
endif()
execute_process(COMMAND ${run_command} RESULT_VARIABLE run_result)
if(NOT run_result EQUAL 0)
  message(FATAL_ERROR "Package consumer execution failed: ${run_result}")
endif()
