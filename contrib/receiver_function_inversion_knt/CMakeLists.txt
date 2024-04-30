# mkdir build; cd build; cmake -DPython_EXECUTABLE=$(which python) ..; make

find_package(
  Python
  COMPONENTS Interpreter Development.Module NumPy
  REQUIRED)

execute_process(
  COMMAND ${Python_EXECUTABLE} -c
          "import numpy; print(numpy.get_include())"
  OUTPUT_VARIABLE NUMPY_INCLUDE_DIRS
  OUTPUT_STRIP_TRAILING_WHITESPACE)

set(module_name "rf")
file(GLOB_RECURSE module_src "${CMAKE_CURRENT_SOURCE_DIR}/rf_lib/src/*.c")
set(module_header "${CMAKE_CURRENT_SOURCE_DIR}/rf_lib/src/knt.h")
set(module_pyx "${CMAKE_CURRENT_SOURCE_DIR}/rf_lib/${module_name}.pyx")
set(module_c "${CMAKE_CURRENT_SOURCE_DIR}/rf_lib/${module_name}.c")

add_custom_command(
    OUTPUT ${module_c}
    COMMENT
        "Making ${CMAKE_CURRENT_BINARY_DIR}/rf_lib/rf.c from ${CMAKE_CURRENT_SOURCE_DIR}/rf_lib/rf.pyx"
    COMMAND ${Python_EXECUTABLE} -m cython ${module_pyx}
            --output-file ${module_c}
    DEPENDS ${module_pyx} ${module_src} ${module_header} ${NUMPY_INCLUDE_DIRS}
    VERBATIM
)
python_add_library(${module_name} MODULE ${module_c} ${module_src} WITH_SOABI)
target_include_directories(${module_name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/rf_lib/src ${NUMPY_INCLUDE_DIRS})

if(SKBUILD)
    install(TARGETS ${module_name} LIBRARY DESTINATION _receiver_function_inversion_knt/build)
else()
    install(TARGETS ${module_name} LIBRARY DESTINATION .)
endif()    
