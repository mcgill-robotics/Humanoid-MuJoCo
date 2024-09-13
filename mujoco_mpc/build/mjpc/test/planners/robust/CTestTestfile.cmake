# CMake generated Testfile for 
# Source directory: /Humanoid-MPC/mujoco_mpc/mjpc/test/planners/robust
# Build directory: /Humanoid-MPC/mujoco_mpc/build/mjpc/test/planners/robust
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(RobustPlannerTest.RandomSearch "/Humanoid-MPC/mujoco_mpc/build/bin/robust_planner_test" "--gtest_filter=RobustPlannerTest.RandomSearch")
set_tests_properties(RobustPlannerTest.RandomSearch PROPERTIES  WORKING_DIRECTORY "/Humanoid-MPC/mujoco_mpc/mjpc/test" _BACKTRACE_TRIPLES "/usr/share/cmake-3.22/Modules/GoogleTest.cmake;400;add_test;/Humanoid-MPC/mujoco_mpc/mjpc/test/CMakeLists.txt;31;gtest_add_tests;/Humanoid-MPC/mujoco_mpc/mjpc/test/planners/robust/CMakeLists.txt;15;test;/Humanoid-MPC/mujoco_mpc/mjpc/test/planners/robust/CMakeLists.txt;0;")
