################## Compiling Fortran rfc executable ###################################
set(f2py_module_name_fmm "_fmm")
set(fortran_src_file_fmm "${CMAKE_CURRENT_SOURCE_DIR}/fmst/src/fm2dss.f90")
set_source_files_properties(${fortran_src_file_fmm} PROPERTIES LANGUAGE "Fortran")
add_executable("fm2dss" ${fortran_src_file_fmm})
