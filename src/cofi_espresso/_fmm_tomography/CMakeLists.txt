if(SKBUILD)
  message(STATUS "The project is built using scikit-build")
  set(fmm_fortran_src "${CMAKE_CURRENT_SOURCE_DIR}/fmst/src/fm2dss.f90")
  set_source_files_properties(${fmm_fortran_src} PROPERTIES LANGUAGE "Fortran")
  set(fmm_generated "fm2dss.o")
  add_executable(${fmm_generated} ${fmm_fortran_src})
  install(FILES 
    ${CMAKE_CURRENT_BINARY_DIR}/${fmm_generated} 
    ${CMAKE_CURRENT_BINARY_DIR}/globalp.mod
    ${CMAKE_CURRENT_BINARY_DIR}/traveltime.mod
    DESTINATION _fmm_tomography/)
else()
  message(STATUS "The project is NOT built using scikit-build")
  cmake_minimum_required(VERSION 3.11.0)
  project(FMMTOMO Fortran)
  add_executable(fm2dss.o fmst/src/fm2dss.f90)
endif()
