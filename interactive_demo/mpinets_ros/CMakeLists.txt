cmake_minimum_required(VERSION 3.0.2)
project(mpinets_ros)

## Find catkin macros and libraries
find_package(catkin REQUIRED COMPONENTS
    interactive_markers
    roscpp
    rospy
    tf2
    visualization_msgs
)

## Uncomment this if the package has a setup.py. This macro ensures
## modules and global scripts declared therein get installed
## See http://ros.org/doc/api/catkin/html/user_guide/setup_dot_py.html
catkin_python_setup()

###################################
## catkin specific configuration ##
###################################
catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES mpinets_ros
#  CATKIN_DEPENDS other_catkin_pkg
#  DEPENDS system_lib
)

###########
## Build ##
###########

## Specify additional locations of header files
include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)

#############
## Install ##
#############

# Install Python nodes for reactive control system
catkin_install_python(PROGRAMS
    nodes/reactive_controller.py
    nodes/reactive_interaction_node.py
    nodes/pointcloud_processor.py
    DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

## Mark other files for installation (e.g. launch and bag files, etc.)
install(FILES
    launch/visualize.launch
    # Add other launch files if any
    DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)