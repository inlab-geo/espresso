# mkdir build; cd build; cmake -DPython_EXECUTABLE=$(which python) ..; make

find_package(
  Python
  COMPONENTS Interpreter Development.Module NumPy
  REQUIRED)

enable_language(Fortran)

execute_process(
  COMMAND ${Python_EXECUTABLE} -c
          "import numpy.f2py; print(numpy.f2py.get_include())"
  OUTPUT_VARIABLE F2PY_INCLUDE_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE)

add_library(fortranobject OBJECT "${F2PY_INCLUDE_DIR}/fortranobject.c")
target_link_libraries(fortranobject PUBLIC Python::NumPy)
target_include_directories(fortranobject PUBLIC "${F2PY_INCLUDE_DIR}")
set_property(TARGET fortranobject PROPERTY POSITION_INDEPENDENT_CODE ON)

set(module_name "rfc")
set(module_src ${CMAKE_CURRENT_SOURCE_DIR}/RF.f90)
file(GLOB_RECURSE module_src_subs "${CMAKE_CURRENT_SOURCE_DIR}/RFsubs/*.f*")
set(module_c "${CMAKE_CURRENT_BINARY_DIR}/${module_name}module.c")

add_custom_command(
  OUTPUT ${module_c} "${module_name}-f2pywrappers.f"
  COMMAND ${Python_EXECUTABLE} -m numpy.f2py ${module_src} 
          -m ${module_name}
          --lower
  DEPENDS ${module_src} 
  VERBATIM
)

python_add_library(${module_name} MODULE ${module_c} ${module_src_subs} ${module_src} WITH_SOABI)
target_link_libraries(${module_name} PRIVATE fortranobject)

if(SKBUILD)
  install(TARGETS ${module_name} DESTINATION __receiver_function_inversion_shibutani/build)
else()
  install(TARGETS ${module_name} DESTINATION .)
endif()
