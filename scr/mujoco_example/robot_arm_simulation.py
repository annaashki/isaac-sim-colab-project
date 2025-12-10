import time

import glfw  # type: ignore[import-not-found]
import mujoco  # type: ignore[import-not-found]
import mujoco.viewer  # type: ignore[import-not-found]

print("Creating 6-axis robot arm model...1")

# MuJoCo XML model definition for 6-axis robot arm
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
      <freejoint/>
      <geom name="cube_geom" type="box" size="0.05 0.05 0.05" rgba="0.9 0.7 0.2 1"/>
      <inertial pos="0 0 0" mass="0.2" diaginertia="0.000167 0.000167 0.000167"/>
    </body>
    
    <!-- Robot Base -->
    <body name="base" pos="0 0 0.1">
      <geom name="base_geom" type="cylinder" size="0.1 0.1" rgba="0.5 0.5 0.5 1"/>
      <inertial pos="0 0 0" mass="2.0" diaginertia="0.01 0.01 0.01"/>
      
      <!-- Joint 6: Base rotation (spins the arm) -->
      <joint name="base_rotation" type="hinge" axis="0 0 1" range="-3.14159 3.14159" damping="12.0" armature="0.1"/>
      
      <!-- Link 1 - vertical arm -->
      <body name="link1" pos="0 0 0.25">
        <geom name="link1_geom" type="cylinder" size="0.08 0.15" rgba="0.2 0.4 0.8 1"/>
        <inertial pos="0 0 0" mass="1.0" diaginertia="0.01 0.01 0.01"/>
        
        <!-- Joint 5: First elbow joint -->
        <joint name="elbow1" type="hinge" axis="1 0 0" range="-1.57 1.57" damping="12.0" armature="0.1"/>
        
        <!-- Link 2 - horizontal arm extending forward -->
        <body name="link2" pos="0 0 0.2">
          <geom name="link2_geom" type="capsule" fromto="0 0 0 0 0.4 0" size="0.05" rgba="0.8 0.3 0.3 1"/>
          <inertial pos="0 0.2 0" mass="0.8" diaginertia="0.008 0.008 0.008"/>
          
          <!-- Joint 4: Second elbow joint -->
          <joint name="elbow2" type="hinge" axis="1 0 0" range="0 3.14159" damping="12.0" armature="0.1"/>
          
          <!-- Link 3 - another horizontal segment -->
          <body name="link3" pos="0 0.4 0">
            <geom name="link3_geom" type="capsule" fromto="0 0 0 0 0.3 0" size="0.04" rgba="0.3 0.8 0.3 1"/>
            <inertial pos="0 0.15 0" mass="0.8" diaginertia="0.008 0.008 0.008"/>
            
            <!-- Joint 3: Third elbow joint -->
            <joint name="elbow3" type="hinge" axis="1 0 0" range="-3.14159 3.14159" damping="8.0" armature="0.05"/>
            
            <!-- Link 4 (gripper mount) - wrist -->
            <body name="link4" pos="0 0.3 0">
              <geom name="link4_geom" type="capsule" fromto="0 0 0 0 0.08 0" size="0.025" rgba="0.3 0.3 0.8 1"/>
              <inertial pos="0 0.04 0" mass="0.3" diaginertia="0.003 0.003 0.003"/>
              
              <!-- Joint 2: Gripper rotation -->
              <joint name="gripper_rotation" type="hinge" axis="0 1 0" range="-3.14159 3.14159" damping="6.0" armature="0.02"/>
              
              <!-- Gripper fingers -->
              <body name="gripper_left" pos="0 0.15 0.05">
                <geom name="gripper_left_geom" type="box" size="0.015 0.08 0.01" rgba="0.9 0.1 0.1 1"/>
                <inertial pos="0 0 0" mass="0.05" diaginertia="0.001 0.001 0.001"/>
                
                <!-- Joint 1: Gripper (left finger) -->
                <joint name="gripper_left" type="slide" axis="0 0 1" range="-0.03 0.03" damping="4.0" armature="0.01"/>
              </body>
              
              <body name="gripper_right" pos="0 0.15 -0.05">
                <geom name="gripper_right_geom" type="box" size="0.015 0.08 0.01" rgba="0.9 0.1 0.1 1"/>
                <inertial pos="0 0 0" mass="0.05" diaginertia="0.001 0.001 0.001"/>
                
                <!-- Gripper (right finger - mirrors left) -->
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

# Create model and data
model = mujoco.MjModel.from_xml_string(xml_string)
data = mujoco.MjData(model)

# Get joint and actuator indices
joint_ids: dict[str, int] = {name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
             for name in ['base_rotation', 'elbow1', 'elbow2', 'elbow3', 'gripper_rotation', 'gripper_left', 'gripper_right']}

actuator_ids: dict[str, int] = {name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
                for name in ['base_rot_actuator', 'elbow1_actuator', 'elbow2_actuator', 
                            'elbow3_actuator', 'gripper_rot_actuator', 'gripper_left_actuator', 'gripper_right_actuator']}

# Joint control targets - initialize in vertical position
joint_targets: dict[str, float] = {
  'base_rotation': 0.0,
  'elbow1': -0.75,  # tilt shoulder so upper arm leans back slightly
  'elbow2': 1.57,   # straight position (middle of 0 to π range, can bend ±90°)
  'elbow3': -0.60,  # orient wrist so gripper points roughly downward
  'gripper_rotation': 0.0,
  'gripper': 0.0  # Controls both gripper fingers
}

joint_ranges: dict[str, tuple[float, float]] = {
  'base_rotation': (-3.14159, 3.14159),
  'elbow1': (-1.57, 1.57),
  'elbow2': (0, 3.14159),
  'elbow3': (-1.57, 1.57),
  'gripper_rotation': (-3.14159, 3.14159),
  'gripper': (-0.03, 0.03)
}


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


for name, limits in joint_ranges.items():
    joint_targets[name] = clamp(joint_targets[name], *limits)

# Set initial joint positions to match the upright pose
data.qpos[joint_ids['base_rotation']] = joint_targets['base_rotation']
data.qpos[joint_ids['elbow1']] = joint_targets['elbow1']
data.qpos[joint_ids['elbow2']] = joint_targets['elbow2']
data.qpos[joint_ids['elbow3']] = joint_targets['elbow3']

# Ensure derived quantities match the imposed configuration
mujoco.mj_forward(model, data)

print("Robot arm created successfully!")
print("\nControls (hold keys for smooth motion):")
print("- Q/A: Base rotation")
print("- W/S: Elbow 1")
print("- E/D: Elbow 2")
print("- R/F: Elbow 3")
print("- T/G: Gripper rotation")
print("- Y/H: Gripper open/close")
print("- ESC: Exit")

# Initialize actuator controls so simulation starts from the target pose
data.ctrl[actuator_ids['base_rot_actuator']] = joint_targets['base_rotation']
data.ctrl[actuator_ids['elbow1_actuator']] = joint_targets['elbow1']
data.ctrl[actuator_ids['elbow2_actuator']] = joint_targets['elbow2']
data.ctrl[actuator_ids['elbow3_actuator']] = joint_targets['elbow3']
data.ctrl[actuator_ids['gripper_rot_actuator']] = joint_targets['gripper_rotation']
data.ctrl[actuator_ids['gripper_left_actuator']] = joint_targets['gripper']
data.ctrl[actuator_ids['gripper_right_actuator']] = -joint_targets['gripper']

# Dictionary to track which keys are currently held down
keys_held: dict[int, bool] = {}

# Key bindings for continuous control (positive key, negative key, rate per second)
continuous_bindings: list[tuple[str, int, int, float]] = [
  ('base_rotation', glfw.KEY_Q, glfw.KEY_A, 0.8),
  ('elbow1', glfw.KEY_W, glfw.KEY_S, 0.9),
  ('elbow2', glfw.KEY_E, glfw.KEY_D, 0.9),
  ('elbow3', glfw.KEY_R, glfw.KEY_F, 1.1),
  ('gripper_rotation', glfw.KEY_T, glfw.KEY_G, 1.2),
  ('gripper', glfw.KEY_Y, glfw.KEY_H, 0.06),
]

# Create a mapping of each key to its opposite key
key_opposites: dict[int, int] = {}
for _, pos_key, neg_key, _ in continuous_bindings:
    key_opposites[pos_key] = neg_key
    key_opposites[neg_key] = pos_key

# Key callback function - called when keys are pressed/released
def key_callback(keycode: int) -> None:
    # Toggle the pressed key
    if keycode in keys_held:
        keys_held[keycode] = not keys_held[keycode]
    else:
        keys_held[keycode] = True
    
    # If this key is now active, deactivate its opposite
    if keys_held.get(keycode, False) and keycode in key_opposites:
        opposite_key: int = key_opposites[keycode]
        keys_held[opposite_key] = False

# Launch viewer with key callback
with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
    print("\nViewer launched. Press keys to toggle continuous movement.")
    print("Press a key once to start moving, press again to stop.")

    dt: float = model.opt.timestep

    # Simulation loop
    while viewer.is_running():
        # Check for continuous key presses
        for joint, pos_key, neg_key, rate in continuous_bindings:
            direction: int = 0
            if keys_held.get(pos_key, False):
                direction += 1
            if keys_held.get(neg_key, False):
                direction -= 1
            if direction:
                joint_targets[joint] = clamp(
                    joint_targets[joint] + direction * rate * dt,
                    *joint_ranges[joint]
                )

        # Update control targets
        data.ctrl[actuator_ids['base_rot_actuator']] = joint_targets['base_rotation']
        data.ctrl[actuator_ids['elbow1_actuator']] = joint_targets['elbow1']
        data.ctrl[actuator_ids['elbow2_actuator']] = joint_targets['elbow2']
        data.ctrl[actuator_ids['elbow3_actuator']] = joint_targets['elbow3']
        data.ctrl[actuator_ids['gripper_rot_actuator']] = joint_targets['gripper_rotation']
        data.ctrl[actuator_ids['gripper_left_actuator']] = joint_targets['gripper']
        data.ctrl[actuator_ids['gripper_right_actuator']] = -joint_targets['gripper']  # Mirror movement

        # Step simulation
        mujoco.mj_step(model, data)

        # Sync viewer
        viewer.sync()

        # Control loop rate
        time.sleep(dt)

print("\nSimulation ended.")
