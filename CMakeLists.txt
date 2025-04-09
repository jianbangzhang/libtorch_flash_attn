cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(example-app)

set(CMAKE_PREFIX_PATH "/home/whu/libtorch/share/cmake/Torch;${CMAKE_PREFIX_PATH}")

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(example-app example-app.cpp)
target_link_libraries(example-app "${TORCH_LIBRARIES}")
set_property(TARGET example-app PROPERTY CXX_STANDARD 17)

add_executable(flash_atten_main flash_atten_main.cpp flash.cu)
target_link_libraries(flash_atten_main "${TORCH_LIBRARIES}")
set_property(TARGET flash_atten_main PROPERTY CXX_STANDARD 17)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  target_compile_options(flash_atten_main PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
endif()


# The following code block is suggested to be used on Windows.
# file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
#  add_custom_command(TARGET example-app
#                     POST_BUILD
#                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
#                     ${TORCH_DLLS}
#                     $<TARGET_FILE_DIR:example-app>)
# endif (MSVC)
