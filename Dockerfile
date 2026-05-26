FROM osrf/ros:jazzy-desktop

ENV DEBIAN_FRONTEND=noninteractive
ENV ROS_DISTRO=jazzy
ENV TURTLEBOT3_MODEL=burger

RUN apt update && apt install -y \
    python3-pip \
    python3-colcon-common-extensions \
    git \
    nano \
    x11-apps \
    mesa-utils \
    ros-jazzy-ros-gz \
    ros-jazzy-ros-gz-sim \
    ros-jazzy-turtlebot3 \
    ros-jazzy-turtlebot3-msgs \
    ros-jazzy-turtlebot3-gazebo \
    ros-jazzy-rviz2 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /root/robot_ws

COPY ./robot_ws/src ./src

RUN /bin/bash -c "source /opt/ros/jazzy/setup.bash && colcon build || true"

CMD ["/bin/bash"]
