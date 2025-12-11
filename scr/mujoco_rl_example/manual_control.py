"""
Manual control script to test robot arm movement speed and limits.
Use keyboard to command target joint angles.
"""
import numpy as np
import mujoco
import mujoco.viewer
import time

# Create the model
xml_string = """
<mujoco model="robot_arm">
  <compiler angle="radian"/>
  
  <option gravity="0 0 -9.81" timestep="0.002"/>
  
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
    <body name="cube" pos="0.5 0.5 0.5">
      <freejoint name="cube_joint"/>
      <geom name="cube_geom" type="box" size="0.05 0.05 0.05" rgba="0.9 0.7 0.2 1"/>
      <inertial pos="0 0 0" mass="0.2" diaginertia="0.000167 0.000167 0.000167"/>
    </body>
    
    <!-- Robot Base -->
    <body name="base" pos="0 0 0.1">
      <geom name="base_geom" type="cylinder" size="0.1 0.1" rgba="0.5 0.5 0.5 1"/>
      <inertial pos="0 0 0" mass="2.0" diaginertia="0.01 0.01 0.01"/>
      
      <joint name="base_rotation" type="hinge" axis="0 0 1" range="-3.14159 3.14159" damping="12.0" armature="0.1"/>
      
      <body name="link1" pos="0 0 0.25">
        <geom name="link1_geom" type="cylinder" size="0.08 0.15" rgba="0.2 0.4 0.8 1"/>
        <inertial pos="0 0 0" mass="1.0" diaginertia="0.01 0.01 0.01"/>
        
        <joint name="elbow1" type="hinge" axis="1 0 0" range="-1.57 1.57" damping="12.0" armature="0.1"/>
        
        <body name="link2" pos="0 0 0.2">
          <geom name="link2_geom" type="capsule" fromto="0 0 0 0 0.4 0" size="0.05" rgba="0.8 0.3 0.3 1"/>
          <inertial pos="0 0.2 0" mass="0.8" diaginertia="0.008 0.008 0.008"/>
          
          <joint name="elbow2" type="hinge" axis="1 0 0" range="0 3.14159" damping="12.0" armature="0.1"/>
          
          <body name="link3" pos="0 0.4 0">
            <geom name="link3_geom" type="capsule" fromto="0 0 0 0 0.3 0" size="0.04" rgba="0.3 0.8 0.3 1"/>
            <inertial pos="0 0.15 0" mass="0.8" diaginertia="0.008 0.008 0.008"/>
            
            <joint name="elbow3" type="hinge" axis="1 0 0" range="-3.14159 3.14159" damping="8.0" armature="0.05"/>
            
            <body name="link4" pos="0 0.3 0">
              <geom name="link4_geom" type="capsule" fromto="0 0 0 0 0.08 0" size="0.025" rgba="0.3 0.3 0.8 1"/>
              <inertial pos="0 0.04 0" mass="0.3" diaginertia="0.003 0.003 0.003"/>
              
              <joint name="gripper_rotation" type="hinge" axis="0 1 0" range="-3.14159 3.14159" damping="6.0" armature="0.02"/>
              
              <site name="gripper_site" pos="0 0.15 0" size="0.01" rgba="1 0 0 0.5"/>
              
              <body name="gripper_left" pos="0 0.15 0.05">
                <geom name="gripper_left_geom" type="box" size="0.015 0.08 0.01" rgba="0.9 0.1 0.1 1"/>
                <inertial pos="0 0 0" mass="0.05" diaginertia="0.001 0.001 0.001"/>
                
                <joint name="gripper_left" type="slide" axis="0 0 1" range="-0.03 0.03" damping="4.0" armature="0.01"/>
              </body>
              
              <body name="gripper_right" pos="0 0.15 -0.05">
                <geom name="gripper_right_geom" type="box" size="0.015 0.08 0.01" rgba="0.9 0.1 0.1 1"/>
                <inertial pos="0 0 0" mass="0.05" diaginertia="0.001 0.001 0.001"/>
                
                <joint name="gripper_right" type="slide" axis="0 0 1" range="-0.03 0.03" damping="4.0" armature="0.01"/>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
  
  <actuator>
    <position name="base_rot_actuator" joint="base_rotation" kp="280" kv="55" ctrlrange="-3.14159 3.14159"/>
    <position name="elbow1_actuator" joint="elbow1" kp="280" kv="55" ctrlrange="-1.57 1.57"/>
    <position name="elbow2_actuator" joint="elbow2" kp="280" kv="55" ctrlrange="0 3.14159"/>
    <position name="elbow3_actuator" joint="elbow3" kp="230" kv="45" ctrlrange="-3.14159 3.14159"/>
    <position name="gripper_rot_actuator" joint="gripper_rotation" kp="140" kv="30" ctrlrange="-3.14159 3.14159"/>
    <position name="gripper_left_actuator" joint="gripper_left" kp="70" kv="18" ctrlrange="-0.03 0.03"/>
    <position name="gripper_right_actuator" joint="gripper_right" kp="70" kv="18" ctrlrange="-0.03 0.03"/>
  </actuator>
</mujoco>
"""

print("🤖 Manual Robot Arm Control")
print("="*60)
print("\nThis lets you test the robot's movement speed and limits.")
print("\nControls:")
print("  1/2: Base rotation (left/right)")
print("  3/4: Elbow 1 (up/down)")
print("  5/6: Elbow 2 (extend/retract)")
print("  7/8: Elbow 3 (rotate)")
print("  9/0: Gripper rotation")
print("  -/=: Gripper open/close")
print("  R: Reset to home position")
print("  ESC: Exit")
print("\nMovement parameters:")
print(f"  Position controller gains: kp=280, kv=55 (main joints)")
print(f"  Joint damping: 8-12 (resists fast motion)")
print(f"  Timestep: 0.002s (500Hz)")
print("="*60)
print()

# Create model and data
model = mujoco.MjModel.from_xml_string(xml_string)
data = mujoco.MjData(model)

# Joint names and ranges
joint_info = {
    'base_rotation': (-3.14159, 3.14159, 0),
    'elbow1': (-1.57, 1.57, 1),
    'elbow2': (0, 3.14159, 2),
    'elbow3': (-3.14159, 3.14159, 3),
    'gripper_rotation': (-3.14159, 3.14159, 4),
    'gripper_left': (-0.03, 0.03, 5),
    'gripper_right': (-0.03, 0.03, 6)
}

# Home position
home_position = np.array([0.0, -0.75, 1.57, -0.60, 0.0, 0.0, 0.0])

# Set initial position
data.qpos[:7] = home_position
data.ctrl[:7] = home_position

# --- RANDOMIZE CUBE POSITION ON STARTUP ---
def randomize_cube(data, model):
  # Random position in circle (min radius 0.25, max radius 0.5)
  angle = np.random.uniform(0, 2 * np.pi)
  radius = np.random.uniform(0.25, 0.5)
  x = radius * np.cos(angle)
  y = radius * np.sin(angle)
  z = 0.5
  cube_pos = np.array([x, y, z])
  cube_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "cube_joint")
  cube_qpos_addr = model.jnt_qposadr[cube_joint_id]
  data.qpos[cube_qpos_addr:cube_qpos_addr+3] = cube_pos
  data.qpos[cube_qpos_addr+3:cube_qpos_addr+7] = [1, 0, 0, 0]  # identity quaternion
  mujoco.mj_forward(model, data)

randomize_cube(data, model)
mujoco.mj_forward(model, data)

# Target angles (what we're commanding)
target_angles = home_position.copy()

# Movement increment (in radians)
joint_increment = 0.1  # 5.7 degrees per keypress
gripper_increment = 0.005

print(f"✓ Robot initialized at home position")
print(f"✓ Movement increment: {np.degrees(joint_increment):.1f} degrees per keypress")
print(f"\nPress keys to move the robot...\n")

# Track previous joint angles to measure velocity
prev_qpos = data.qpos[:7].copy()
max_velocity = 0.0

with mujoco.viewer.launch_passive(model, data) as viewer:
    step_count = 0
    def handle_key(key):
      # R or r resets robot and randomizes cube
      if key in (ord('R'), ord('r')):
        data.qpos[:7] = home_position
        data.ctrl[:7] = home_position
        randomize_cube(data, model)
        mujoco.mj_forward(model, data)
        print("[INFO] Reset: robot to home, cube randomized.")

    viewer.user_key_callback = handle_key

    while viewer.is_running():
      step_start = time.time()
      # Command the target angles
      data.ctrl[:7] = target_angles
      mujoco.mj_step(model, data)
      current_qpos = data.qpos[:7].copy()
      dt = model.opt.timestep
      velocity = np.abs(current_qpos - prev_qpos) / dt
      max_vel_this_step = np.max(velocity)
      max_velocity = max(max_velocity, max_vel_this_step)
      prev_qpos = current_qpos.copy()
      if step_count % 500 == 0:
        print(f"Step {step_count}")
        print(f"  Target angles: {np.degrees(target_angles)}")
        print(f"  Current angles: {np.degrees(data.qpos[:7])}")
        print(f"  Max velocity seen: {np.degrees(max_velocity):.1f} deg/s")
        print(f"  Position error: {np.degrees(np.abs(target_angles - data.qpos[:7]))}")
        print()
      step_count += 1
      viewer.sync()
      time.sleep(model.opt.timestep)

print("\n" + "="*60)
print("Maximum joint velocity observed: {:.1f} deg/s".format(np.degrees(max_velocity)))
print("="*60)
