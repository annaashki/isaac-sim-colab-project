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
        data.qpos[addr:addr+3] = np.array([0.2, 0.2, 0.5])
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

  #model, data = set_cube_position(model, data, False)

  return model, data

XML_STRING = """
<mujoco model="robot_arm">
  <compiler angle="radian"/>

  <option gravity="0 0 -9.81" timestep="0.001"/>

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

    <!-- Cube: SOLID with reasonable friction -->
    <body name="cube" pos="0.20 0.20 0.05">
      <freejoint name="cube_joint"/>
      <geom name="cube_geom" type="box" size="0.03 0.03 0.03" rgba="0.9 0.7 0.2 1" 
            friction="1.0 0.005 0.0001" solref="0.01 1" solimp="0.9 0.95 0.001"/>
      <inertial pos="0 0 0" mass="0.1" diaginertia="0.000167 0.000167 0.00016"/>
    </body>

    <!-- Robot Base -->
    <body name="base" pos="0 0 0.08">
      <geom name="base_geom" type="cylinder" size="0.10 0.08" rgba="0.5 0.5 0.5 1"/>
      <inertial pos="0 0 0" mass="2.0" diaginertia="0.01 0.01 0.01"/>

      <joint name="base_rotation" type="hinge" axis="0 0 1" range="-3.14159 3.14159" damping="6.0" armature="0.1"/>

      <body name="link1_root" pos="0 0 0.16">
        <joint name="elbow1" type="hinge" axis="1 0 0" range="-1.57 1.57" damping="6.0" armature="0.12"/>

        <geom name="link1_geom" type="capsule" fromto="0 0 0 0 0 0.14" size="0.05" rgba="0.2 0.4 0.8 1"/>
        <inertial pos="0 0 0.07" mass="1.0" diaginertia="0.01 0.01 0.01"/>

        <body name="link2_root" pos="0 0 0.14">
          <joint name="elbow2" type="hinge" axis="1 0 0" range="-2 2" damping="6.0" armature="0.1" ref="-1.57"/>

          <geom name="link2_geom" type="capsule" fromto="0 0 0 0 0.14 0" size="0.05" rgba="0.8 0.3 0.3 1"/>
          <inertial pos="0 0.07 0" mass="0.8" diaginertia="0.008 0.008 0.008"/>

          <body name="link3_root" pos="0 0.14 0">
            <joint name="elbow3" type="hinge" axis="1 0 0" range="-2 2" damping="6.0" armature="0.05"/>

            <geom name="link3_geom" type="capsule" fromto="0 0 0 0 0.12 0" size="0.04" rgba="0.3 0.8 0.3 1"/>
            <inertial pos="0 0.06 0" mass="0.8" diaginertia="0.008 0.008 0.008"/>

            <!-- SHORTENED wrist link: 0.04 instead of 0.08 -->
            <body name="wrist_root" pos="0 0.12 0">
              <joint name="gripper_rotation" type="hinge" axis="0 1 0" range="-2 2" damping="5.0" armature="0.02"/>

              <geom name="link4_geom" type="capsule" fromto="0 0 0 0 0.04 0" size="0.025" rgba="0.3 0.3 0.8 1"/>
              <inertial pos="0 0.02 0" mass="0.3" diaginertia="0.002 0.002 0.002"/>

              <!-- Gripper base at shortened wrist tip -->
              <body name="gripper_base" pos="0 0.04 0">
                <site name="gripper_site" pos="0 0 0" size="0.01" rgba="1 0 0 0.5"/>

                <!-- SOLID gripper fingers: slide along Z, properly positioned -->
                <body name="gripper_left" pos="0 0.08 0.025">
                  <joint name="gripper_left" type="slide" axis="0 0 1" range="0.0 0.025" damping="15.0" armature="0.03"/>
                  <geom name="gripper_left_geom" type="box" size="0.012 0.035 0.008" rgba="0.9 0.1 0.1 1" 
                        friction="1.5 0.005 0.0001" solref="0.01 1" solimp="0.95 0.99 0.001"/>
                  <inertial pos="0 0 0" mass="0.05" diaginertia="0.0008 0.0008 0.0008"/>
                </body>

                <body name="gripper_right" pos="0 0.08 -0.025">
                  <joint name="gripper_right" type="slide" axis="0 0 1" range="-0.025 0.0" damping="15.0" armature="0.03"/>
                  <geom name="gripper_right_geom" type="box" size="0.012 0.035 0.008" rgba="0.9 0.1 0.1 1" 
                        friction="1.5 0.005 0.0001" solref="0.01 1" solimp="0.95 0.99 0.001"/>
                  <inertial pos="0 0 0" mass="0.05" diaginertia="0.0008 0.0008 0.0008"/>
                </body>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>
  </worldbody>

  <equality>
    <joint name="gripper_mirror" joint1="gripper_left" joint2="gripper_right" polycoef="0 -1 0 0 0"/>
  </equality>

  <actuator>
    <position name="base_rot_actuator" joint="base_rotation" kp="100" kv="30" ctrlrange="-3.1415 3.1415"/>
    <position name="elbow1_actuator" joint="elbow1" kp="100" kv="30" ctrlrange="-1.57 1.57"/>
    <position name="elbow2_actuator" joint="elbow2" kp="100" kv="30" ctrlrange="-2 2"/>
    <position name="elbow3_actuator" joint="elbow3" kp="100" kv="30" ctrlrange="-2 2"/>
    <position name="gripper_rot_actuator" joint="gripper_rotation" kp="40" kv="20" ctrlrange="-2 2"/>
    <position name="gripper_actuator" joint="gripper_left" kp="50" kv="25" ctrlrange="-0.025 0.025"/>
  </actuator>
</mujoco>
"""