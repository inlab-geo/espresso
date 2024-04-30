################################################################################
# Project configuration and settings
################################################################################
cmake_minimum_required(VERSION 3.15...3.26)

project(Espresso LANGUAGES C Fortran)

find_package(
  Python
  COMPONENTS Interpreter Development.Module NumPy
  REQUIRED)

################################################################################
# Package content
################################################################################
add_subdirectory(src/espresso/)
