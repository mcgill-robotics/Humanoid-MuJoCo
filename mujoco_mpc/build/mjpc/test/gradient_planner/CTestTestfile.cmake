# CMake generated Testfile for 
# Source directory: /Humanoid-MPC/mujoco_mpc/mjpc/test/gradient_planner
# Build directory: /Humanoid-MPC/mujoco_mpc/build/mjpc/test/gradient_planner
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(GradientPlannerTest.Particle "/Humanoid-MPC/mujoco_mpc/build/bin/gradient_planner_test" "--gtest_filter=GradientPlannerTest.Particle")
set_tests_properties(GradientPlannerTest.Particle PROPERTIES  WORKING_DIRECTORY "/Humanoid-MPC/mujoco_mpc/mjpc/test" _BACKTRACE_TRIPLES "/usr/share/cmake-3.22/Modules/GoogleTest.cmake;400;add_test;/Humanoid-MPC/mujoco_mpc/mjpc/test/CMakeLists.txt;31;gtest_add_tests;/Humanoid-MPC/mujoco_mpc/mjpc/test/gradient_planner/CMakeLists.txt;15;test;/Humanoid-MPC/mujoco_mpc/mjpc/test/gradient_planner/CMakeLists.txt;0;")
add_test(GradientTest.Gradient "/Humanoid-MPC/mujoco_mpc/build/bin/gradient_test" "--gtest_filter=GradientTest.Gradient")
set_tests_properties(GradientTest.Gradient PROPERTIES  WORKING_DIRECTORY "/Humanoid-MPC/mujoco_mpc/mjpc/test" _BACKTRACE_TRIPLES "/usr/share/cmake-3.22/Modules/GoogleTest.cmake;400;add_test;/Humanoid-MPC/mujoco_mpc/mjpc/test/CMakeLists.txt;31;gtest_add_tests;/Humanoid-MPC/mujoco_mpc/mjpc/test/gradient_planner/CMakeLists.txt;18;test;/Humanoid-MPC/mujoco_mpc/mjpc/test/gradient_planner/CMakeLists.txt;0;")
add_test(GradientTest.ZeroTest "/Humanoid-MPC/mujoco_mpc/build/bin/zero_test" "--gtest_filter=GradientTest.ZeroTest")
set_tests_properties(GradientTest.ZeroTest PROPERTIES  WORKING_DIRECTORY "/Humanoid-MPC/mujoco_mpc/mjpc/test" _BACKTRACE_TRIPLES "/usr/share/cmake-3.22/Modules/GoogleTest.cmake;400;add_test;/Humanoid-MPC/mujoco_mpc/mjpc/test/CMakeLists.txt;31;gtest_add_tests;/Humanoid-MPC/mujoco_mpc/mjpc/test/gradient_planner/CMakeLists.txt;25;test;/Humanoid-MPC/mujoco_mpc/mjpc/test/gradient_planner/CMakeLists.txt;0;")
add_test(GradientTest.LinearTest "/Humanoid-MPC/mujoco_mpc/build/bin/linear_test" "--gtest_filter=GradientTest.LinearTest")
set_tests_properties(GradientTest.LinearTest PROPERTIES  WORKING_DIRECTORY "/Humanoid-MPC/mujoco_mpc/mjpc/test" _BACKTRACE_TRIPLES "/usr/share/cmake-3.22/Modules/GoogleTest.cmake;400;add_test;/Humanoid-MPC/mujoco_mpc/mjpc/test/CMakeLists.txt;31;gtest_add_tests;/Humanoid-MPC/mujoco_mpc/mjpc/test/gradient_planner/CMakeLists.txt;28;test;/Humanoid-MPC/mujoco_mpc/mjpc/test/gradient_planner/CMakeLists.txt;0;")
add_test(GradientTest.CubicTest "/Humanoid-MPC/mujoco_mpc/build/bin/cubic_test" "--gtest_filter=GradientTest.CubicTest")
set_tests_properties(GradientTest.CubicTest PROPERTIES  WORKING_DIRECTORY "/Humanoid-MPC/mujoco_mpc/mjpc/test" _BACKTRACE_TRIPLES "/usr/share/cmake-3.22/Modules/GoogleTest.cmake;400;add_test;/Humanoid-MPC/mujoco_mpc/mjpc/test/CMakeLists.txt;31;gtest_add_tests;/Humanoid-MPC/mujoco_mpc/mjpc/test/gradient_planner/CMakeLists.txt;31;test;/Humanoid-MPC/mujoco_mpc/mjpc/test/gradient_planner/CMakeLists.txt;0;")
