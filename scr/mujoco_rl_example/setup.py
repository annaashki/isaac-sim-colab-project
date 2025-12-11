import numpy as np
import mujoco
from mujoco import MjModel, MjData

def set_cube_position(model, data, randomize:bool=True):
   # Find the freejoint address for the cube
    j_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'cube_joint')
    addr = model.jnt_qposadr[j_id]  # start index into qpos for this joint

    # Free joint has 7 qpos entries: [x, y, z, qw, qx, qy, qz]
    # Randomize x,y,z within some bounds; keep quaternion = identity
    if not randomize:
        # Set to default position
        data.qpos[addr:addr+3] = np.array([0.3, 0.3, 0.7])
        data.qpos[addr+3:addr+7] = np.array([1.0, 0.0, 0.0, 0.0])  # identity orientation
        mujoco.mj_forward(model, data)
        return model, data
    
    xy_bounds = np.array([[0.3, 0.5],   # x min/max
                          [-0.5, 0.5]])  # y min/max
    z_bounds = [0.4, 0.6]             # z min/max

    x = np.random.uniform(*xy_bounds[0])
    y = np.random.uniform(*xy_bounds[1])
    z = np.random.uniform(*z_bounds)

    data.qpos[addr:addr+3] = np.array([x, y, z])
    print(f"Cube position is: {x}, {y}, {z}")

    def random_quaternion():
        u1, u2, u3 = np.random.rand(3)
        q = np.array([
            np.sqrt(1-u1) * np.sin(2*np.pi*u2),
            np.sqrt(1-u1) * np.cos(2*np.pi*u2),
            np.sqrt(u1)   * np.sin(2*np.pi*u3),
            np.sqrt(u1)   * np.cos(2*np.pi*u3),
        ])
        # Return in MuJoCo order [w, x, y, z]
        return np.array([q[3], q[0], q[1], q[2]])
    data.qpos[addr+3:addr+7] = random_quaternion()
    # Optional: zero velocity
    data.qvel[addr:addr+6] = 0.0
    mujoco.mj_forward(model, data) # identity orientation
    return model, data
   

def gen_model_and_data():
  model = MjModel.from_xml_string(XML_STRING)
  data = MjData(model)

  model, data = set_cube_position(model, data, False)

  return model, data

XML_STRING = """
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
        <geom name="floor" type="plane" size="5 5 0.1" rgba="0.9 0.9 0.9 1" material="grid"/>
        <light pos="0 0 3" dir="0 0 -1" diffuse="1 1 1"/>
        
        <!-- Cube to pick up -->
        <body name="cube" pos="0.5 0.5 0.7">
          <freejoint name="cube_joint"/>
          <geom name="cube_geom" type="box" size="0.05 0.05 0.05" rgba="0.9 0.7 0.2 1"/> # Non-collidable remove contype/conaffinity if collisions desired
          <inertial pos="0 0 0" mass="0.2" diaginertia="0.000167 0.000167 0.00016"/>
        </body>
        
        <!-- Robot Base at height 0.08 -->
        <body name="base" pos="0 0 0.08">
        <geom name="base_geom" type="cylinder" size="0.10 0.08" rgba="0.5 0.5 0.5 1"/>
        <inertial pos="0 0 0" mass="2.0" diaginertia="0.01 0.01 0.01"/>

        <!-- Base rotation at base COM -->
        <joint name="base_rotation" type="hinge" axis="0 0 1" range="-3.14159 3.14159" damping="5.0" armature="0.1"/>

        <!-- Link1 starts at the TOP of the base: z = 0.09 -->
        <body name="link1_root" pos="0 0 0.08">
          <!-- Elbow1 hinge located at link1 root (top of base) -->
          <joint name="elbow1" type="hinge" axis="1 0 0" range="-1.57 1.57" damping="5.0" armature="0.1"/>

          <!-- Link1: 0.14 along +z -->
          <geom name="link1_geom" type="capsule" fromto="0 0 0 0 0 0.14" size="0.05" rgba="0.2 0.4 0.8 1"/>
          <inertial pos="0 0 0.07" mass="1.0" diaginertia="0.01 0.01 0.01"/>

          <!-- Link2 root at the end of Link1: z = 0.08 + 0.14 = 0.22 -->
          <body name="link2_root" pos="0 0 0.14">
            <joint name="elbow2" type="hinge" axis="1 0 0" range="-2 2" damping="5.0" armature="0.1" ref="-1.57"/>

            <!-- Link2: 0.14 along +y -->
            <geom name="link2_geom" type="capsule" fromto="0 0 0 0 0.14 0" size="0.05" rgba="0.8 0.3 0.3 1"/>
            <inertial pos="0 0.07 0" mass="0.8" diaginertia="0.008 0.008 0.008"/>

            <!-- Link3 root at end of Link2: y = 0.14 -->
            <body name="link3_root" pos="0 0.14 0">
              <joint name="elbow3" type="hinge" axis="1 0 0" range="-2 2" damping="5.0" armature="0.05"/>

              <!-- Link3: 0.14 along +y -->
              <geom name="link3_geom" type="capsule" fromto="0 0 0 0 0.14 0" size="0.04" rgba="0.3 0.8 0.3 1"/>
              <inertial pos="0 0.07 0" mass="0.8" diaginertia="0.008 0.008 0.008"/>

              <!-- Wrist + gripper mount: remaining 0.08 along +y to reach total 0.20 from start of link3 -->
              <body name="wrist_root" pos="0 0.12 0">
                <joint name="gripper_rotation" type="hinge" axis="0 1 0" range="-2 2" damping="5.0" armature="0.02"/>

                <geom name="link4_geom" type="capsule" fromto="0 0 0 0 0.08 0" size="0.025" rgba="0.3 0.3 0.8 1"/>
                <inertial pos="0 0.04 0" mass="0.5" diaginertia="0.003 0.003 0.003"/>

                <!-- End-effector site at wrist tip -->
                <site name="gripper_site" pos="0 0.08 0" size="0.01" rgba="1 0 0 0.5"/>

                <!-- Gripper fingers centered at wrist tip, sliding along z -->
                <body name="gripper_left" pos="0 0.12 0.03">
                  <joint name="gripper_left" type="slide" axis="0 0 1" range="-0.03 0.03" damping="5.0" armature="0.01"/>
                  <geom name="gripper_left_geom" type="box" size="0.015 0.04 0.01" rgba="0.9 0.1 0.1 1" friction="3 0.005 0.0001"/>
                  <inertial pos="0 0 0" mass="0.05" diaginertia="0.001 0.001 0.001"/>
                </body>

                <body name="gripper_right" pos="0 0.12 -0.03">
                  <joint name="gripper_right" type="slide" axis="0 0 1" range="-0.03 0.03" damping="5.0" armature="0.01"/>
                  <geom name="gripper_right_geom" type="box" size="0.015 0.04 0.01" rgba="0.9 0.1 0.1 1" friction="20 0.005 0.0001"/>
                  <inertial pos="0 0 0" mass="0.05" diaginertia="0.001 0.001 0.001"/>
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
        <position name="base_rot_actuator" joint="base_rotation" kp="100" kv="30" ctrlrange="-3.1415 3.1415"/>
        <position name="elbow1_actuator" joint="elbow1" kp="100" kv="30" ctrlrange="-1.57 1.57"/>
        <position name="elbow2_actuator" joint="elbow2" kp="100" kv="30" ctrlrange="-2 2"/>
        <position name="elbow3_actuator" joint="elbow3" kp="100" kv="30" ctrlrange="-2 2"/>
        <position name="gripper_rot_actuator" joint="gripper_rotation" kp="100" kv="30" ctrlrange="-2 2"/>
        <position name="gripper_actuator" joint="gripper_left" kp="100" kv="30" ctrlrange="-0.03 0.03"/>
      </actuator>
    </mujoco>
    """
    