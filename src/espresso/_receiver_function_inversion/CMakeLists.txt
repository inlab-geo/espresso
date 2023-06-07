### setup project ###
cmake_minimum_required(VERSION 3.9)

project(rfc C Fortran)

# Safety net
if(PROJECT_SOURCE_DIR STREQUAL PROJECT_BINARY_DIR)
  message(
    FATAL_ERROR
      "In-source builds not allowed. Please make a new directory (called a build directory) and run CMake from there.\n"
  )
endif()

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

# scikit-build style includes
find_package(PythonExtensions REQUIRED) # for ${PYTHON_EXTENSION_MODULE_SUFFIX}

# Grab the variables from a local Python installation
# NumPy headers
execute_process(
  COMMAND "${PYTHON_EXECUTABLE}"
  -c "import numpy; print(numpy.get_include())"
  OUTPUT_VARIABLE NumPy_INCLUDE_DIRS
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
# F2PY headers
execute_process(
  COMMAND "${PYTHON_EXECUTABLE}"
  -c "import numpy.f2py; print(numpy.f2py.get_include())"
  OUTPUT_VARIABLE F2PY_INCLUDE_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Prepping the module
set(f2py_module_name "rfc")
set(fortran_src_file ${CMAKE_CURRENT_SOURCE_DIR}/RF.f90)
file(GLOB_RECURSE fortran_src_file_subs "${CMAKE_CURRENT_SOURCE_DIR}/RFsubs/*.f*")
set(f2py_module_c ${f2py_module_name}module.c)
file(TO_CMAKE_PATH "${F2PY_INCLUDE_DIR}/fortranobject.c" f2py_fortran_object)

# Target for enforcing dependencies
add_custom_command(
  OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/${f2py_module_c}"
  COMMAND ${PYTHON_EXECUTABLE}  -m "numpy.f2py"
                   "${fortran_src_file}"
                   -m "${f2py_module_name}"
                   --lower # Important
  DEPENDS ${fortran_src_file} # Fortran source
)

add_library(${f2py_module_name} MODULE
            "${f2py_module_c}"
            "${f2py_fortran_object}"
            "${fortran_src_file_subs}"
            "${fortran_src_file}")

target_include_directories(${f2py_module_name} PUBLIC
                           ${F2PY_INCLUDE_DIR}
                           ${NumPy_INCLUDE_DIRS}
                           ${PYTHON_INCLUDE_DIRS})
set_target_properties(${f2py_module_name} PROPERTIES SUFFIX "${PYTHON_EXTENSION_MODULE_SUFFIX}")
set_target_properties(${f2py_module_name} PROPERTIES PREFIX "")

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
  install(TARGETS ${f2py_module_name} DESTINATION _receiver_function_inversion/build)
else()
  install(TARGETS ${f2py_module_name} DESTINATION .)
endif()
