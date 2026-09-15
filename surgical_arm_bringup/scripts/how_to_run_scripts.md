# Instructions on how to run the scripts on the robot

## 1. Insertion.py
# Start the gen3_complete.launch.py after sshing into the robot
```
ros2 launch surgical_arm_bringup gen3_complete_system.launch.py
 ```

Run the insertion script (standard 90 degree insertion)
``` 
ros2 run surgical_arm_bringup insertion.py --ros-args -p real_robot:=true execute_motion:=true -p target_x:=0.5 -p target_y:=-0.2 -p target_z:=0.45 -p insertion_angle_deg:=45.0 -p target_depth_mm:=30.0 -p skip_home:=false ```
```

Run the insertion script (standard 45 degree insertion)
```
ros2 run surgical_arm_bringup insertion.py --ros-args -p real_robot:=true -p execute_motion:=true -p target_x:=0.45 -p target_y:=-0.20 -p target_depth_mm:=30.0 -p insertion_angle_deg:=45.0 -p step_by_step:=false -p skip_home_move:=false -p direct_to_angled_hover:=true
```

## 2. Impedance control 

## 3. RealSense eye-to-hand calibration

See [realsense_handeye_calibration.md](realsense_handeye_calibration.md) for the
four-terminal procedure, the frame-name trap when publishing the result, and
validation.
