"""
Gymnasium environment for robot arm pick-and-place task with MuJoCo.
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco


class RobotArmPickEnv(gym.Env):
    """Custom Environment for robot arm to pick up a cube."""
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}
    
    # MuJoCo XML model definition for 6-axis robot arm
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
                  
                  <!-- Gripper site for end-effector position -->
                  <site name="gripper_site" pos="0 0.15 0" size="0.01" rgba="1 0 0 0.5"/>
                  
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
    
    def __init__(self, render_mode=None, max_episode_steps=500):
        super().__init__()
        
        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps
        self.current_step = 0
        
        # Create MuJoCo model and data
        self.model = mujoco.MjModel.from_xml_string(self.XML_STRING)
        self.data = mujoco.MjData(self.model)
        
        # Initialize viewer if rendering
        self.viewer = None
        
        # Get important IDs
        self.cube_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "cube")
        self.gripper_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "gripper_site")
        
        # Joint IDs (7 actuated joints)
        self.joint_names = ['base_rotation', 'elbow1', 'elbow2', 'elbow3', 
                           'gripper_rotation', 'gripper_left', 'gripper_right']
        
        # Action space: 7 continuous actions (one per joint)
        # Normalized to [-1, 1]
        self.action_space = spaces.Box(
            low=-1.0, 
            high=1.0, 
            shape=(7,), 
            dtype=np.float32
        )
        
        # Observation space:
        # - 7 joint positions
        # - 7 joint velocities
        # - 3 gripper position (x, y, z)
        # - 3 cube position (x, y, z)
        # - 3 relative position (cube - gripper)
        # Total: 23 features
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(23,),
            dtype=np.float32
        )
        
        # Joint ranges for action scaling
        self.joint_ranges = {
            'base_rotation': (-3.14159, 3.14159),
            'elbow1': (-1.57, 1.57),
            'elbow2': (0, 3.14159),
            'elbow3': (-3.14159, 3.14159),
            'gripper_rotation': (-3.14159, 3.14159),
            'gripper_left': (-0.03, 0.03),
            'gripper_right': (-0.03, 0.03)
        }
        
        # Initial cube position for randomization
        self.initial_cube_pos = np.array([0.5, 0.5, 0.5])
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Reset simulation
        mujoco.mj_resetData(self.model, self.data)
        self.current_step = 0
        
        # Set initial robot pose
        initial_qpos = {
            'base_rotation': 0.0,
            'elbow1': -0.75,
            'elbow2': 1.57,
            'elbow3': -0.60,
            'gripper_rotation': 0.0,
            'gripper_left': 0.0,
            'gripper_right': 0.0
        }
        
        for i, joint_name in enumerate(self.joint_names):
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            self.data.qpos[joint_id] = initial_qpos[joint_name]
            self.data.ctrl[i] = initial_qpos[joint_name]
        
        # Randomize cube position slightly
        if self.np_random is not None:
            cube_pos = self.initial_cube_pos + self.np_random.uniform(-0.15, 0.15, size=3)
            cube_pos[2] = 0.5  # Keep height constant
        else:
            cube_pos = self.initial_cube_pos.copy()
        
        # Set cube position (freejoint has 7 DOF: 3 pos + 4 quat)
        cube_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "cube_joint")
        cube_qpos_addr = self.model.jnt_qposadr[cube_joint_id]
        self.data.qpos[cube_qpos_addr:cube_qpos_addr+3] = cube_pos
        # Set quaternion to identity (w=1, x=0, y=0, z=0)
        self.data.qpos[cube_qpos_addr+3:cube_qpos_addr+7] = [1, 0, 0, 0]
        
        # Step forward to stabilize
        mujoco.mj_forward(self.model, self.data)
        
        observation = self._get_obs()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action):
        # Scale actions from [-1, 1] to actual joint ranges
        scaled_action = np.zeros(7)
        for i, joint_name in enumerate(self.joint_names):
            low, high = self.joint_ranges[joint_name]
            # Scale from [-1, 1] to [low, high]
            scaled_action[i] = low + (action[i] + 1.0) * 0.5 * (high - low)
        
        # Apply action
        self.data.ctrl[:7] = scaled_action
        
        # Step simulation (multiple substeps for stability)
        for _ in range(10):  # 10 substeps per action
            mujoco.mj_step(self.model, self.data)
        
        self.current_step += 1
        
        # Get observation and info
        observation = self._get_obs()
        info = self._get_info()
        
        # Calculate reward
        reward = self._compute_reward(info)
        
        # Check termination
        terminated = info['is_success']
        truncated = self.current_step >= self.max_episode_steps
        
        # Render if needed
        if self.render_mode == "human":
            self.render()
        
        return observation, reward, terminated, truncated, info
    
    def _get_obs(self):
        # Get joint positions and velocities (first 7 joints)
        qpos = self.data.qpos[:7].copy()
        qvel = self.data.qvel[:7].copy()
        
        # Get gripper position
        gripper_pos = self.data.site_xpos[self.gripper_site_id].copy()
        
        # Get cube position (from freejoint)
        cube_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "cube_joint")
        cube_qpos_addr = self.model.jnt_qposadr[cube_joint_id]
        cube_pos = self.data.qpos[cube_qpos_addr:cube_qpos_addr+3].copy()
        
        # Relative position
        relative_pos = cube_pos - gripper_pos
        
        # Concatenate all observations
        obs = np.concatenate([qpos, qvel, gripper_pos, cube_pos, relative_pos])
        
        return obs.astype(np.float32)
    
    def _get_info(self):
        # Get gripper and cube positions
        gripper_pos = self.data.site_xpos[self.gripper_site_id].copy()
        cube_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "cube_joint")
        cube_qpos_addr = self.model.jnt_qposadr[cube_joint_id]
        cube_pos = self.data.qpos[cube_qpos_addr:cube_qpos_addr+3].copy()
        
        # Calculate distance
        distance = np.linalg.norm(cube_pos - gripper_pos)
        
        # Check if cube is picked up (gripper close to cube and cube is lifted)
        cube_height = cube_pos[2]
        gripper_distance = distance
        
        # Success criteria: cube lifted above 0.7m and gripper is close
        is_success = cube_height > 0.7 and gripper_distance < 0.15
        
        return {
            'distance_to_cube': distance,
            'cube_height': cube_height,
            'is_success': is_success,
            'gripper_pos': gripper_pos,
            'cube_pos': cube_pos
        }
    
    def _compute_reward(self, info):
        """
        Reward function for pick-and-place task:
        - Dense reward based on distance to cube
        - Bonus for getting close to cube
        - Large bonus for lifting the cube
        - Success bonus for completing task
        """
        distance = info['distance_to_cube']
        cube_height = info['cube_height']
        
        # Distance reward (negative, so smaller distance = higher reward)
        distance_reward = -distance * 2.0
        
        # Close proximity bonus
        if distance < 0.1:
            distance_reward += 1.0
        
        # Lifting reward
        if cube_height > 0.6:
            lift_reward = (cube_height - 0.5) * 10.0
        else:
            lift_reward = 0.0
        
        # Success bonus
        success_reward = 100.0 if info['is_success'] else 0.0
        
        total_reward = distance_reward + lift_reward + success_reward
        
        return total_reward
    
    def render(self):
        if self.render_mode == "human":
            if self.viewer is None:
                try:
                    import mujoco.viewer
                    self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
                except (AttributeError, ImportError):
                    # Fallback: viewer not available, skip rendering
                    print("Warning: MuJoCo viewer not available, skipping visualization")
                    self.render_mode = None
                    return
            if self.viewer is not None:
                self.viewer.sync()
    
    def close(self):
        if self.viewer is not None:
            try:
                self.viewer.close()
            except:
                pass
            self.viewer = None
