# CMake generated Testfile for 
# Source directory: /Humanoid-MPC/mujoco_mpc/mjpc/test/sampling_planner
# Build directory: /Humanoid-MPC/mujoco_mpc/build/mjpc/test/sampling_planner
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(SamplingPlannerTest.RandomSearch "/Humanoid-MPC/mujoco_mpc/build/bin/sampling_planner_test" "--gtest_filter=SamplingPlannerTest.RandomSearch")
set_tests_properties(SamplingPlannerTest.RandomSearch PROPERTIES  WORKING_DIRECTORY "/Humanoid-MPC/mujoco_mpc/mjpc/test" _BACKTRACE_TRIPLES "/usr/share/cmake-3.22/Modules/GoogleTest.cmake;400;add_test;/Humanoid-MPC/mujoco_mpc/mjpc/test/CMakeLists.txt;31;gtest_add_tests;/Humanoid-MPC/mujoco_mpc/mjpc/test/sampling_planner/CMakeLists.txt;15;test;/Humanoid-MPC/mujoco_mpc/mjpc/test/sampling_planner/CMakeLists.txt;0;")
