include(FetchContent)
cmake_minimum_required(VERSION 3.11.0)
project(espresso_swmp)

FetchContent_Declare(
  swmp
  GIT_REPOSITORY https://github.com/JuergHauser/swmp.git
  #GIT_REPOSITORY /home/jrh/csiro/research/latent/multipathing/swmp.git
  GIT_TAG main
)

FetchContent_MakeAvailable(swmp)

