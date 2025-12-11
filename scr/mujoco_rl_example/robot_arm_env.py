"""
Gymnasium environment for robot arm pick-and-place task with MuJoCo.
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco

print("Initializing RobotArmPickEnv...543")

class RobotArmPickEnv(gym.Env):
    """Custom Environment for robot arm to pick up a cube."""
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}
    
    # MuJoCo XML model definition for 6-axis robot arm
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
    
    def __init__(self, render_mode=None, max_episode_steps=500):
        super().__init__()
        
        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps  # More steps for better learning
        self.current_step = 0
        
        # Create MuJoCo model and data
        self.model = mujoco.MjModel.from_xml_string(self.XML_STRING)
        self.data = mujoco.MjData(self.model)
        
        # Initialize viewer if rendering
        self.viewer = None
        
        # Get important IDs
        self.cube_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "cube")
        self.gripper_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "gripper_site")
        
        # Joint IDs (6 actuated joints - gripper fingers mirror each other)
        self.joint_names = ['base_rotation', 'elbow1', 'elbow2', 'elbow3', 
                           'gripper_rotation', 'gripper_left']
        
        # Action space: 6 target joint angles (normalized to [-1, 1])
        # Robot will move to these angles over multiple simulation steps
        # This matches real robot control where you command target angles
        self.action_space = spaces.Box(
            low=-1.0, 
            high=1.0, 
            shape=(6,), 
            dtype=np.float32
        )
        
        # Number of simulation steps to reach target (makes movement smoother)
        self.action_repeat = 1000  # ~2.0 seconds - reasonable time for smooth robot motion
        
        # Observation space:
        # - 6 joint positions
        # - 6 joint velocities
        # - 3 gripper position (x, y, z)
        # - 3 cube position (x, y, z)
        # - 3 relative position (cube - gripper)
        # Total: 21 features
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(21,),
            dtype=np.float32
        )
        
        # Joint ranges for action scaling
        self.joint_ranges = {
            'base_rotation': (-3.14159, 3.14159),
            'elbow1': (-1.57, 1.57),
            'elbow2': (-1.57, 1.57),
            'elbow3': (-1.57, 1.57),
            'gripper_rotation': (-1.57, 1.57),
            'gripper_left': (-0.03, 0.03)
        }
        
        # Initial cube position for randomization
        self.initial_cube_pos = np.array([0.5, 0.5, 0.5])
        
        # No need for additional PID control - MuJoCo's position actuators handle this
        # The kp and kv gains in the XML are sufficient
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Reseed random generator for proper cube randomization
        self.np_random, _ = gym.utils.seeding.np_random(seed)
        
        # Reset simulation
        mujoco.mj_resetData(self.model, self.data)
        self.current_step = 0
        
        # Set initial robot pose to match manual_control.py home position
        initial_qpos = {
          'base_rotation': 0.0,
          'elbow1': -0.75,
          'elbow2': 1.57,
          'elbow3': -0.60,
          'gripper_rotation': 0.0,
          'gripper_left': 0.0
        }
        
        for i, joint_name in enumerate(self.joint_names):
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            self.data.qpos[joint_id] = initial_qpos[joint_name]
            self.data.ctrl[i] = initial_qpos[joint_name]
        
        # Initialize gripper_right to mirror gripper_left (will be handled by equality constraint)
        gripper_right_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "gripper_right")
        self.data.qpos[gripper_right_id] = 0.0
        
        # Randomize cube position within a circle (min radius 0.25, max radius 0.5)
        if self.np_random is not None:
          angle = self.np_random.uniform(0, 2 * np.pi)
          radius = self.np_random.uniform(0.25, 0.5)
          x = radius * np.cos(angle)
          y = radius * np.sin(angle)
          z = 0.5  # Keep height constant
          cube_pos = np.array([x, y, z])
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
        # Scale actions from [-1, 1] to actual joint angle ranges
        # This is the target joint configuration the robot should move to
        target_angles = np.zeros(6)
        for i, joint_name in enumerate(self.joint_names):
            low, high = self.joint_ranges[joint_name]
            # Scale from [-1, 1] to [low, high]
            target_angles[i] = low + (action[i] + 1.0) * 0.5 * (high - low)
        
        # Set target angles for the position actuators
        # MuJoCo's position actuators (with kp and kv gains) will smoothly move to target
        self.data.ctrl[:6] = target_angles
        
        # Step simulation for action_repeat steps (2 seconds of sim time)
        for i in range(self.action_repeat):
            mujoco.mj_step(self.model, self.data)
            
            # Sync viewer every 25 steps for smooth visualization
            if self.viewer is not None and i % 25 == 0:
                self.viewer.sync()
        
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
        # Get joint positions and velocities (first 6 controlled joints)
        qpos = self.data.qpos[:6].copy()
        qvel = self.data.qvel[:6].copy()
        
        # Get gripper position
        gripper_pos = self.data.site_xpos[self.gripper_site_id].copy()
        
        # Get cube position (from freejoint)
        cube_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "cube_joint")
        cube_qpos_addr = self.model.jnt_qposadr[cube_joint_id]
        cube_pos = self.data.qpos[cube_qpos_addr:cube_qpos_addr+3].copy()
        
        # Relative position
        relative_pos = cube_pos - gripper_pos
        
        # Concatenate all observations (6+6+3+3+3 = 21)
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
        Improved reward function for pick-and-place task.
        Provides dense, shaped rewards to guide learning.
        """
        distance = info['distance_to_cube']
        cube_height = info['cube_height']
        initial_cube_height = 0.5  # Initial z position of cube
        
        # 1. Approaching reward: Strong exponential reward for getting close
        # Scale: 10.0 when very close, ~0.5 at 1m away
        approach_reward = 10.0 * np.exp(-2.0 * distance)
        
        # 2. Contact/proximity bonus: Extra reward for being very close
        if distance < 0.15:
            contact_bonus = 5.0 * (1.0 - distance / 0.15)  # 5.0 bonus when touching, 0 at 0.15m
        else:
            contact_bonus = 0.0
        
        # 3. Lifting reward: Strong reward for lifting the cube
        height_improvement = cube_height - initial_cube_height
        if height_improvement > 0.02:  # Cube moved up even slightly
            lift_reward = height_improvement * 50.0  # Strong reward for lifting
        else:
            lift_reward = 0.0
        
        # 4. Height-maintaining reward: Extra reward for keeping cube elevated
        if cube_height > 0.6:  # Cube is above initial position
            height_bonus = 10.0 * (cube_height - 0.6)
        else:
            height_bonus = 0.0
        
        # 5. Success bonus: Large reward for completing the task
        success_reward = 100.0 if info['is_success'] else 0.0
        
        # 6. Small penalty for high joint velocities to encourage smooth motion
        joint_velocities = self.data.qvel[:6]
        velocity_penalty = -0.01 * np.sum(np.square(joint_velocities))
        
        # Combine all rewards
        total_reward = (approach_reward + contact_bonus + lift_reward + 
                       height_bonus + success_reward + velocity_penalty)
        
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
