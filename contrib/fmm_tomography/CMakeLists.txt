################## Compiling Fortran rfc executable ###################################
set(fmm_module_name "_fmm")

set(fmm_fortran_src "${CMAKE_CURRENT_SOURCE_DIR}/fmst/src/fm2dss.f90")
set_source_files_properties(${fmm_fortran_src} PROPERTIES LANGUAGE "Fortran")

set(fmm_generated "fm2dss")

add_executable(${fmm_generated} ${fmm_fortran_src})
install(FILES 
    ${CMAKE_CURRENT_BINARY_DIR}/${fmm_generated} 
    ${CMAKE_CURRENT_BINARY_DIR}/globalp.mod
    ${CMAKE_CURRENT_BINARY_DIR}/traveltime.mod
    DESTINATION fmm_tomography/)
