import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco
from scr.mujoco_rl_example.setup import gen_model_and_data

class RobotArmPickEnv(gym.Env):
    """Custom Environment for robot arm to pick up a cube."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self, render_mode=None, max_episode_steps=500):
        super().__init__()
        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps
        self.current_step = 0

        # Initialize model and data (randomizes cube)
        self.model, self.data = gen_model_and_data()

        # Initialize viewer if rendering
        self.viewer = None

        # IDs
        self.cube_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "cube")
        self.gripper_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "gripper_site")

        # Joint names (6 actuated joints)
        self.joint_names = [
            "base_rotation", "elbow1", "elbow2", "elbow3",
            "gripper_rotation", "gripper_left"
        ]

        # Action space: 6 normalized joint targets
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)

        # Simulation steps per action
        self.action_repeat = 1000

        # Observation: 21 features
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(21,), dtype=np.float32
        )

        # Joint ranges for scaling actions
        self.joint_ranges = {
            "base_rotation": (-3.14159, 3.14159),
            "elbow1": (-1.57, 1.57),
            "elbow2": (-2.0, 2.0),
            "elbow3": (-2.0, 12.0),
            "gripper_rotation": (-2.0, 2.0),
            "gripper_left": (-0.03, 0.03),
        }

        self.initial_cube_pos = np.array([0.5, 0.5, 0.5], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Gymnasium seeding helper
        self.np_random, _ = gym.utils.seeding.np_random(seed)
        self.current_step = 0

        # Re-initialize model and data (randomizes cube)
        self.model, self.data = gen_model_and_data()

        # Recompute IDs after re-init
        self.cube_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "cube")
        self.gripper_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "gripper_site")

        # Set initial robot pose
        initial_qpos = {
            "base_rotation": 0.0,
            "elbow1": 0.0,
            "elbow2": 0.0,
            "elbow3": 0.0,
            "gripper_rotation": 0.0,
            "gripper_left": 0.0,
        }
        for i, joint_name in enumerate(self.joint_names):
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            self.data.qpos[joint_id] = initial_qpos[joint_name]
            self.data.ctrl[i] = initial_qpos[joint_name]

        # Mirror right gripper via equality constraint; still set initial value explicitly
        gripper_right_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "gripper_right")
        if gripper_right_id != -1:
            self.data.qpos[gripper_right_id] = 0.0

        mujoco.mj_forward(self.model, self.data)

        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def step(self, action):
        # Validate action dtype/shape
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (6,):
            raise ValueError(f"Action must have shape (6,), got {action.shape}")

        # Scale actions from [-1, 1] to joint ranges
        target_angles = np.zeros(6, dtype=np.float32)
        for i, joint_name in enumerate(self.joint_names):
            low, high = self.joint_ranges[joint_name]
            target_angles[i] = low + (action[i] + 1.0) * 0.5 * (high - low)

        # Apply targets to position actuators
        self.data.ctrl[:6] = target_angles

        # Simulate
        for i in range(self.action_repeat):
            mujoco.mj_step(self.model, self.data)
            if self.viewer is not None and i % 25 == 0:
                self.viewer.sync()

        self.current_step += 1

        observation = self._get_obs()
        info = self._get_info()
        reward = self._compute_reward(info)

        terminated = bool(info["is_success"])
        truncated = bool(self.current_step >= self.max_episode_steps)

        if self.render_mode == "human":
            self.render()

        return observation, reward, terminated, truncated, info

    def _get_obs(self):
        # Joint states
        qpos = self.data.qpos[:6].copy()
        qvel = self.data.qvel[:6].copy()

        # Gripper position
        gripper_pos = self.data.site_xpos[self.gripper_site_id].copy()

        # Cube position from free joint
        cube_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "cube_joint")
        cube_qpos_addr = self.model.jnt_qposadr[cube_joint_id]
        cube_pos = self.data.qpos[cube_qpos_addr : cube_qpos_addr + 3].copy()

        # Relative position
        relative_pos = cube_pos - gripper_pos

        obs = np.concatenate([qpos, qvel, gripper_pos, cube_pos, relative_pos]).astype(np.float32)
        return obs

    def _get_info(self):
        gripper_pos = self.data.site_xpos[self.gripper_site_id].copy()
        cube_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "cube_joint")
        cube_qpos_addr = self.model.jnt_qposadr[cube_joint_id]
        cube_pos = self.data.qpos[cube_qpos_addr : cube_qpos_addr + 3].copy()

        distance = np.linalg.norm(cube_pos - gripper_pos)
        cube_height = float(cube_pos[2])

        # Success: lifted and close
        is_success = bool(cube_height > 0.7 and distance < 0.15)

        return {
            "distance_to_cube": float(distance),
            "cube_height": cube_height,
            "is_success": is_success,
            "gripper_pos": gripper_pos.copy(),
            "cube_pos": cube_pos.copy(),
        }

    def _compute_reward(self, info):
        distance = info["distance_to_cube"]
        cube_height = info["cube_height"]
        initial_cube_height = 0.5

        approach_reward = 10.0 * np.exp(-2.0 * distance)
        contact_bonus = 5.0 * (1.0 - distance / 0.15) if distance < 0.15 else 0.0

        height_improvement = cube_height - initial_cube_height
        lift_reward = 50.0 * max(0.0, height_improvement - 0.02)

        height_bonus = 10.0 * max(0.0, cube_height - 0.6)

        success_reward = 100.0 if info["is_success"] else 0.0

        joint_velocities = self.data.qvel[:6]
        velocity_penalty = -0.01 * float(np.sum(np.square(joint_velocities)))

        total_reward = (
            approach_reward + contact_bonus + lift_reward + height_bonus + success_reward + velocity_penalty
        )
        return float(total_reward)

    def render(self):
        if self.render_mode == "human":
            if self.viewer is None:
                try:
                    import mujoco.viewer
                    self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
                except (AttributeError, ImportError):
                    print("Warning: MuJoCo viewer not available, skipping visualization")
                    self.render_mode = None
                    return
            self.viewer.sync()

    def close(self):
        if self.viewer is not None:
            try:
                self.viewer.close()
            except Exception:
                pass
            self.viewer = None