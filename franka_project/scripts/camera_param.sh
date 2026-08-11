source /opt/ros/jazzy/setup.bash
for CAM in /camera1/camera1 /camera2/camera2; do
    ros2 param set $CAM rgb_camera.exposure 166
    ros2 param set $CAM rgb_camera.gain 64
    ros2 param set $CAM rgb_camera.auto_exposure_priority false
    ros2 param set $CAM rgb_camera.white_balance 4600.0
    ros2 param set $CAM rgb_camera.enable_auto_white_balance true
    ros2 param set $CAM rgb_camera.enable_auto_exposure true
done

## 指定远程屏幕
# export DISPLAY=:0
# export XAUTHORITY=/run/user/1000/gdm/Xauthority
# export XDG_RUNTIME_DIR=/run/user/1000