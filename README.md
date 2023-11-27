# ros2-stereo-camera-analyse
This program uses a stereo camera to detect the burr and the height of the annealed component with the help of ROS2 and rc_visard.

the rc_genicam_api and ros2 are mandatory for this workspace.\
the code can run with:\
	-ros2 run rc_genicam_driver rc_genicam_driver --ros-args -p "device:=:02940786"\
	-ros2 run my_image_processing image_node\
	
Moreover, a ROS2 workspace and package must be created and the source must be set.
