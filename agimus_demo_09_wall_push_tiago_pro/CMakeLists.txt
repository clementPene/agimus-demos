cmake_minimum_required(VERSION 3.22.1)

project(agimus_demo_09_wall_push_tiago_pro LANGUAGES CXX)

find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()

ament_auto_package(INSTALL_TO_SHARE config hpp launch scripts urdf)
