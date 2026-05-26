#!/bin/bash

xhost +local:docker

docker run -it --rm \
  --name multi_robot_sim \
  --net=host \
  --ipc=host \
  -e DISPLAY=$DISPLAY \
  -e QT_X11_NO_MITSHM=1 \
  -e TURTLEBOT3_MODEL=burger \
  -e GZ_SIM_RESOURCE_PATH=/opt/ros/jazzy/share/turtlebot3_gazebo/models:/root/robot_ws/src/follower_controller/models \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v $(pwd)/robot_ws:/root/robot_ws \
  multi-robot-sim
