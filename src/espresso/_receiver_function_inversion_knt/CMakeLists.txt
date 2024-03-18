# mkdir build; cd build; cmake -DPYTHON_EXECUTABLE=$(which python) ..; make

cmake_minimum_required(VERSION 3.9)

project(rf_c)

# Ensure scikit-build modules
if (NOT SKBUILD)
  find_package(PythonInterp 3.8 REQUIRED)
  # Kanged --> https://github.com/Kitware/torch_liberator/blob/master/CMakeLists.txt
  # If skbuild is not the driver; include its utilities in CMAKE_MODULE_PATH
  execute_process(
    COMMAND "${PYTHON_EXECUTABLE}"
    -c "import os, skbuild; print(os.path.dirname(skbuild.__file__))"
    OUTPUT_VARIABLE SKBLD_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  list(APPEND CMAKE_MODULE_PATH "${SKBLD_DIR}/resources/cmake")
  message(STATUS "Looking in ${SKBLD_DIR}/resources/cmake for CMake modules")
endif()

# Grab the variables from a local Python installation
# NumPy headers
execute_process(
  COMMAND "${PYTHON_EXECUTABLE}"
  -c "import numpy; print(numpy.get_include())"
  OUTPUT_VARIABLE NumPy_INCLUDE_DIRS
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

find_package(PythonExtensions REQUIRED)
find_package(Cython REQUIRED)

set(rf_src_files
    rf_lib/src/my_complex.c
    rf_lib/src/fft.c
    rf_lib/src/knt_mini.c
    rf_lib/src/matrix.c
    rf_lib/src/rcvFn.c
)
add_library(rf_lib STATIC ${rf_src_files})
set_target_properties(rf_lib PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(rf_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/rf_lib/src ${NumPy_INCLUDE_DIRS})

add_cython_target(rf rf_lib/rf.pyx)
add_library(rf MODULE ${rf})
target_link_libraries(rf rf_lib)
target_include_directories(rf PRIVATE ${NumPy_INCLUDE_DIRS})
python_extension_module(rf)

# Linker fixes
if (UNIX)
  if (APPLE)
    set_target_properties(${f2py_module_name} PROPERTIES
    LINK_FLAGS  '-Wl,-dylib,-undefined,dynamic_lookup')
  else()
    set_target_properties(${f2py_module_name} PROPERTIES
  LINK_FLAGS  '-Wl,--allow-shlib-undefined')
  endif()
endif()

if(SKBUILD)
    install(TARGETS rf LIBRARY DESTINATION _receiver_function_inversion_knt/build)
else()
    install(TARGETS rf LIBRARY DESTINATION .)
endif()    
