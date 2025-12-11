import mujoco
import mujoco.viewer
import numpy as np

XML_STRING = """
<mujoco model="robot_arm">
  <compiler angle="radian"/>
  
  <option gravity="0 0 0" timestep="0.002"/>
  
  <visual>
    <headlight ambient="0.5 0.5 0.5"/>
  </visual>
  
  <asset>
    <texture name="grid" type="2d" builtin="checker" width="512" height="512" rgb1="0.2 0.3 0.4" rgb2="0.3 0.4 0.5"/>
    <material name="grid" texture="grid" texrepeat="1 1" texuniform="true" reflectance="0.2"/>
  </asset>
  
  <worldbody>
    <!-- Ground plane -->
    <geom name="floor" type="plane" size="2 2 0.1" rgba="0.9 0.9 0.9 1" material="grid"/>
    <light pos="0 0 3" dir="0 0 -1" diffuse="1 1 1"/>
  
    <!-- Cube to pick up -->
    <body name="cube" pos="0.5 0.5 0.7">
      <freejoint name="cube_joint"/>
      <geom name="cube_geom" type="box" size="0.05 0.05 0.05" rgba="0.9 0.7 0.2 1"/>
      <inertial pos="0 0 0" mass="0.2" diaginertia="0.000167 0.000167 0.000167"/>
    </body>
  
    <!-- Robot Base -->
    <body name="base" pos="0 0 0.1">
      <geom name="base_geom" type="cylinder" size="0.1 0.1" rgba="0.5 0.5 0.5 1"/>
      <inertial pos="0 0 0" mass="2.0" diaginertia="0.01 0.01 0.01"/>
  
      <!-- Joint 6: Base rotation (spins the arm) -->
      <joint name="base_rotation" type="hinge" axis="0 0 1" range="-3.14159 3.14159" damping="50.0" armature="0.1" ref="0"/>
  
      <!-- Link 1 - vertical arm -->
      <body name="link1" pos="0 0 0.25">
        <geom name="link1_geom" type="cylinder" size="0.08 0.15" rgba="0.2 0.4 0.8 1"/>
        <inertial pos="0 0 0" mass="1.0" diaginertia="0.01 0.01 0.01"/>
  
      <!-- Joint 1: First elbow joint -->
      <joint name="elbow1" type="hinge" axis="1 0 0" range="-1.57 1.57" damping="50.0" armature="0.1" ref="0"/>
  
        <!-- Link 2 - horizontal arm extending forward -->
        <body name="link2" pos="0 0 0.2">
          <geom name="link2_geom" type="capsule" fromto="0 0 0 0 0.4 0" size="0.05" rgba="0.8 0.3 0.3 1"/>
          <inertial pos="0 0.2 0" mass="0.8" diaginertia="0.008 0.008 0.008"/>
  
          <!-- Joint 4: Second elbow joint -->
          <joint name="elbow2" type="hinge" axis="1 0 0" range="-1.57 1.57" damping="50.0" armature="0.1" ref="-1.57"/>
  
          <!-- Link 3 - another horizontal segment -->
          <body name="link3" pos="0 0.4 0">
            <geom name="link3_geom" type="capsule" fromto="0 0 0 0 0.3 0" size="0.04" rgba="0.3 0.8 0.3 1"/>
            <inertial pos="0 0.15 0" mass="0.8" diaginertia="0.008 0.008 0.008"/>
  
            <!-- Joint 3: Third elbow joint -->
            <joint name="elbow3" type="hinge" axis="1 0 0" range="-1.57 1.57" damping="50.0" armature="0.05" ref="0"/>
  
            <!-- Link 4 (gripper mount) - wrist -->
            <body name="link4" pos="0 0.3 0">
              <geom name="link4_geom" type="capsule" fromto="0 0 0 0 0.08 0" size="0.025" rgba="0.3 0.3 0.8 1"/>
              <inertial pos="0 0.04 0" mass="0.5" diaginertia="0.003 0.003 0.003"/>
  
              <!-- Joint 2: Gripper rotation -->
              <joint name="gripper_rotation" type="hinge" axis="0 1 0" range="-1.57 1.57" damping="50.0" armature="0.02" ref="0"/>
  
              <!-- Gripper site for end-effector position -->
              <site name="gripper_site" pos="0 0.15 0" size="0.01" rgba="1 0 0 0.5"/>
  
              <!-- Gripper fingers -->
              <body name="gripper_left" pos="0 0.15 0.05">
                <geom name="gripper_left_geom" type="box" size="0.015 0.08 0.01" rgba="0.9 0.1 0.1 1"/>
                <inertial pos="0 0 0" mass="0.5" diaginertia="0.001 0.001 0.001"/>
  
                <!-- Joint 1: Gripper (left finger) -->
                <joint name="gripper_left" type="slide" axis="0 0 1" range="-0.03 0.03" damping="50.0" armature="0.01" ref="0"/>
              </body>
  
              <body name="gripper_right" pos="0 0.15 -0.05">
                <geom name="gripper_right_geom" type="box" size="0.015 0.08 0.01" rgba="0.9 0.1 0.1 1"/>
                <inertial pos="0 0 0" mass="0.05" diaginertia="0.001 0.001 0.001"/>
  
                <!-- Gripper (right finger - mirrors left) -->
                <joint name="gripper_right" type="slide" axis="0 0 1" range="-0.03 0.03" damping="50.0" armature="0.01" ref="0"/>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
  
  <equality>
    <!-- Mirror gripper fingers: right moves opposite to left -->
    <joint name="gripper_mirror" joint1="gripper_left" joint2="gripper_right" polycoef="0 -1 0 0 0"/>
  </equality>
  
  <actuator>
    <position name="base_rot_actuator" joint="base_rotation" kp="40" kv="20" ctrlrange="-3.1415 3.1415"/>
    <position name="elbow1_actuator" joint="elbow1" kp="40" kv="20" ctrlrange="-1.57 1.57"/>
    <position name="elbow2_actuator" joint="elbow2" kp="40" kv="20" ctrlrange="-1.57 1.57"/>
    <position name="elbow3_actuator" joint="elbow3" kp="40" kv="20" ctrlrange="-1.57 1.57"/>
    <position name="gripper_rot_actuator" joint="gripper_rotation" kp="40" kv="20" ctrlrange="-1.57 1.57"/>
    <position name="gripper_actuator" joint="gripper_left" kp="40" kv="20" ctrlrange="-0.03 0.03"/>
  </actuator>
</mujoco>
"""

def main():
    """Launch interactive MuJoCo simulation with slider controls."""
    
    # Load model from XML string
    model = mujoco.MjModel.from_xml_string(XML_STRING)
    data = mujoco.MjData(model)
    
    # Initialize control values to zeros (neutral position)
    data.ctrl[:] = 0.0
    
    # Launch the interactive viewer with built-in sliders
    # The viewer automatically creates sliders for all actuators
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Simulation loop
        while viewer.is_running():
            # Step the simulation forward
            mujoco.mj_step(model, data)
            
            # Sync the viewer (updates visualization)
            viewer.sync()

if __name__ == "__main__":
    main()