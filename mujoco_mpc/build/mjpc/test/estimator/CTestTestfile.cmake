# CMake generated Testfile for 
# Source directory: /Humanoid-MPC/mujoco_mpc/mjpc/test/estimator
# Build directory: /Humanoid-MPC/mujoco_mpc/build/mjpc/test/estimator
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(BatchFilter.Box3Drot "/Humanoid-MPC/mujoco_mpc/build/bin/batch_filter_test" "--gtest_filter=BatchFilter.Box3Drot")
set_tests_properties(BatchFilter.Box3Drot PROPERTIES  WORKING_DIRECTORY "/Humanoid-MPC/mujoco_mpc/mjpc/test" _BACKTRACE_TRIPLES "/usr/share/cmake-3.22/Modules/GoogleTest.cmake;400;add_test;/Humanoid-MPC/mujoco_mpc/mjpc/test/CMakeLists.txt;31;gtest_add_tests;/Humanoid-MPC/mujoco_mpc/mjpc/test/estimator/CMakeLists.txt;15;test;/Humanoid-MPC/mujoco_mpc/mjpc/test/estimator/CMakeLists.txt;0;")
add_test(PriorCost.Particle "/Humanoid-MPC/mujoco_mpc/build/bin/batch_prior_test" "--gtest_filter=PriorCost.Particle")
set_tests_properties(PriorCost.Particle PROPERTIES  WORKING_DIRECTORY "/Humanoid-MPC/mujoco_mpc/mjpc/test" _BACKTRACE_TRIPLES "/usr/share/cmake-3.22/Modules/GoogleTest.cmake;400;add_test;/Humanoid-MPC/mujoco_mpc/mjpc/test/CMakeLists.txt;31;gtest_add_tests;/Humanoid-MPC/mujoco_mpc/mjpc/test/estimator/CMakeLists.txt;18;test;/Humanoid-MPC/mujoco_mpc/mjpc/test/estimator/CMakeLists.txt;0;")
add_test(PriorCost.Box "/Humanoid-MPC/mujoco_mpc/build/bin/batch_prior_test" "--gtest_filter=PriorCost.Box")
set_tests_properties(PriorCost.Box PROPERTIES  WORKING_DIRECTORY "/Humanoid-MPC/mujoco_mpc/mjpc/test" _BACKTRACE_TRIPLES "/usr/share/cmake-3.22/Modules/GoogleTest.cmake;400;add_test;/Humanoid-MPC/mujoco_mpc/mjpc/test/CMakeLists.txt;31;gtest_add_tests;/Humanoid-MPC/mujoco_mpc/mjpc/test/estimator/CMakeLists.txt;18;test;/Humanoid-MPC/mujoco_mpc/mjpc/test/estimator/CMakeLists.txt;0;")
add_test(Estimator.Kalman "/Humanoid-MPC/mujoco_mpc/build/bin/kalman_test" "--gtest_filter=Estimator.Kalman")
set_tests_properties(Estimator.Kalman PROPERTIES  WORKING_DIRECTORY "/Humanoid-MPC/mujoco_mpc/mjpc/test" _BACKTRACE_TRIPLES "/usr/share/cmake-3.22/Modules/GoogleTest.cmake;400;add_test;/Humanoid-MPC/mujoco_mpc/mjpc/test/CMakeLists.txt;31;gtest_add_tests;/Humanoid-MPC/mujoco_mpc/mjpc/test/estimator/CMakeLists.txt;21;test;/Humanoid-MPC/mujoco_mpc/mjpc/test/estimator/CMakeLists.txt;0;")
add_test(Unscented.Particle1D "/Humanoid-MPC/mujoco_mpc/build/bin/unscented_test" "--gtest_filter=Unscented.Particle1D")
set_tests_properties(Unscented.Particle1D PROPERTIES  WORKING_DIRECTORY "/Humanoid-MPC/mujoco_mpc/mjpc/test" _BACKTRACE_TRIPLES "/usr/share/cmake-3.22/Modules/GoogleTest.cmake;400;add_test;/Humanoid-MPC/mujoco_mpc/mjpc/test/CMakeLists.txt;31;gtest_add_tests;/Humanoid-MPC/mujoco_mpc/mjpc/test/estimator/CMakeLists.txt;24;test;/Humanoid-MPC/mujoco_mpc/mjpc/test/estimator/CMakeLists.txt;0;")
add_test(Unscented.Box3Drot "/Humanoid-MPC/mujoco_mpc/build/bin/unscented_test" "--gtest_filter=Unscented.Box3Drot")
set_tests_properties(Unscented.Box3Drot PROPERTIES  WORKING_DIRECTORY "/Humanoid-MPC/mujoco_mpc/mjpc/test" _BACKTRACE_TRIPLES "/usr/share/cmake-3.22/Modules/GoogleTest.cmake;400;add_test;/Humanoid-MPC/mujoco_mpc/mjpc/test/CMakeLists.txt;31;gtest_add_tests;/Humanoid-MPC/mujoco_mpc/mjpc/test/estimator/CMakeLists.txt;24;test;/Humanoid-MPC/mujoco_mpc/mjpc/test/estimator/CMakeLists.txt;0;")
