# CMake generated Testfile for 
# Source directory: /Humanoid-MPC/mujoco_mpc/mjpc/test/tasks
# Build directory: /Humanoid-MPC/mujoco_mpc/build/mjpc/test/tasks
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(TasksTest.Task "/Humanoid-MPC/mujoco_mpc/build/bin/task_test" "--gtest_filter=TasksTest.Task")
set_tests_properties(TasksTest.Task PROPERTIES  WORKING_DIRECTORY "/Humanoid-MPC/mujoco_mpc/mjpc/test" _BACKTRACE_TRIPLES "/usr/share/cmake-3.22/Modules/GoogleTest.cmake;400;add_test;/Humanoid-MPC/mujoco_mpc/mjpc/test/CMakeLists.txt;31;gtest_add_tests;/Humanoid-MPC/mujoco_mpc/mjpc/test/tasks/CMakeLists.txt;15;test;/Humanoid-MPC/mujoco_mpc/mjpc/test/tasks/CMakeLists.txt;0;")
add_test(StepAllTasksTest.Task "/Humanoid-MPC/mujoco_mpc/build/bin/task_test" "--gtest_filter=StepAllTasksTest.Task")
set_tests_properties(StepAllTasksTest.Task PROPERTIES  WORKING_DIRECTORY "/Humanoid-MPC/mujoco_mpc/mjpc/test" _BACKTRACE_TRIPLES "/usr/share/cmake-3.22/Modules/GoogleTest.cmake;400;add_test;/Humanoid-MPC/mujoco_mpc/mjpc/test/CMakeLists.txt;31;gtest_add_tests;/Humanoid-MPC/mujoco_mpc/mjpc/test/tasks/CMakeLists.txt;15;test;/Humanoid-MPC/mujoco_mpc/mjpc/test/tasks/CMakeLists.txt;0;")
