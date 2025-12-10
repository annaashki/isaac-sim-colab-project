# Robot Arm Pick-and-Place with Reinforcement Learning

This project demonstrates how to train a 6-axis robot arm to pick up a cube using **Reinforcement Learning (RL)** with MuJoCo and Stable-Baselines3.

## 📋 Overview

- **Environment**: Custom Gymnasium environment wrapping MuJoCo simulation
- **Task**: Pick up a cube and lift it above a target height
- **Algorithms**: PPO (Proximal Policy Optimization) or SAC (Soft Actor-Critic)
- **Observation Space**: 23-dimensional (joint positions/velocities, gripper position, cube position, relative distance)
- **Action Space**: 7-dimensional continuous actions (one per joint)

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install gymnasium stable-baselines3[extra] tensorboard
```

### 2. Train the Agent

**Option A: Standard Training (Fastest, No Visualization)**

```bash
python train.py --algorithm PPO --timesteps 500000
```

**Option B: Training with Periodic Visualization (Watch Progress!)**

```bash
python train_with_rendering.py --timesteps 500000 --eval-freq 10000
```

This shows the robot every 10k steps during evaluation so you can see improvement!

Train using SAC (more sample efficient but slower):

```bash
python train.py --algorithm SAC --timesteps 300000
```

**Training arguments:**
- `--algorithm`: Choose "PPO" or "SAC" (default: PPO)
- `--timesteps`: Total training steps (default: 500,000)
- `--eval-freq`: Show visualization every N steps (train_with_rendering.py only)
- `--save-dir`: Where to save models (default: ./models)
- `--log-dir`: Where to save logs (default: ./logs)

### 3. Monitor Training

Open TensorBoard to monitor training progress:

```bash
tensorboard --logdir ./logs
```

Then open http://localhost:6006 in your browser.

### 4. Test the Trained Agent

Test the trained model with visualization:

```bash
python test.py --episodes 5
```

**Testing arguments:**
- `--model-path`: Path to specific model (auto-finds best if not specified)
- `--models-dir`: Directory to search for models (default: ./models)
- `--algorithm`: Algorithm used (default: PPO)
- `--episodes`: Number of test episodes (default: 5)
- `--no-render`: Run without visualization

## 📁 Project Structure

```
mujoco_rl_example/
├── robot_arm_env.py    # Custom Gymnasium environment
├── train.py            # Training script
├── test.py             # Testing/inference script
├── README.md           # This file
├── models/             # Saved models (created during training)
│   └── PPO_YYYYMMDD_HHMMSS/
│       ├── best_model/
│       ├── final_model/
│       └── checkpoints/
└── logs/               # TensorBoard logs (created during training)
```

## 🎯 How It Works

### Environment (`robot_arm_env.py`)

**Observation Space (23 dimensions):**
- Joint positions (7)
- Joint velocities (7)
- Gripper position in 3D (3)
- Cube position in 3D (3)
- Relative position (cube - gripper) (3)

**Action Space (7 dimensions):**
- Continuous actions for each joint (normalized to [-1, 1])
- Actions are scaled to actual joint ranges

**Reward Function:**
- Distance reward: Encourages gripper to approach cube
- Proximity bonus: Extra reward for getting very close
- Lifting reward: Rewards lifting the cube higher
- Success bonus: Large reward (100) for completing the task

**Success Criteria:**
- Cube lifted above 0.7m
- Gripper within 0.15m of cube

### Training Process

1. **Initialization**: Agent starts with random policy
2. **Exploration**: Agent tries different actions to discover what works
3. **Learning**: Algorithm (PPO/SAC) updates policy based on rewards
4. **Exploitation**: Agent increasingly uses learned strategies
5. **Convergence**: Performance stabilizes as optimal policy is found

**Expected Training Time:**
- PPO: ~30-60 minutes on modern CPU (500k steps)
- SAC: ~45-90 minutes on modern CPU (300k steps)

**Success Indicators:**
- Episode reward increases over time
- Success rate approaches 80-100%
- Agent learns to: approach cube → grasp → lift

## 🔧 Customization

### Modify Reward Function

Edit `_compute_reward()` in `robot_arm_env.py`:

```python
def _compute_reward(self, info):
    # Your custom reward logic here
    distance = info['distance_to_cube']
    cube_height = info['cube_height']
    
    reward = -distance + cube_height * 5.0
    return reward
```

### Change Task Difficulty

Adjust success criteria in `_get_info()`:

```python
# Make it harder (lift higher, closer gripper)
is_success = cube_height > 0.9 and gripper_distance < 0.10

# Make it easier (lift lower, further gripper)
is_success = cube_height > 0.6 and gripper_distance < 0.20
```

### Tune Hyperparameters

Modify parameters in `train.py`:

```python
model = PPO(
    "MlpPolicy",
    env,
    learning_rate=3e-4,      # Learning speed
    n_steps=2048,            # Steps per update
    batch_size=64,           # Batch size
    ent_coef=0.01,           # Exploration coefficient
    # ... other params
)
```

## 📊 Results Interpretation

### TensorBoard Metrics

- `rollout/ep_rew_mean`: Average episode reward (should increase)
- `rollout/ep_len_mean`: Average episode length
- `rollout/success_rate`: Task success rate (goal: >80%)
- `train/loss`: Training loss (should decrease)

### Good Training Signs
✓ Reward steadily increases
✓ Success rate improves
✓ Episode length decreases (finds solution faster)

### Bad Training Signs
✗ Reward stays flat or decreases
✗ High variance in rewards
✗ Success rate stays at 0%

**Solutions:**
- Train longer
- Adjust reward function
- Tune hyperparameters
- Simplify task

## 🐛 Troubleshooting

**Problem: Training is too slow**
- Reduce `total_timesteps`
- Use fewer `n_steps` in PPO
- Try SAC (more sample efficient)

**Problem: Agent doesn't learn**
- Check reward function is not too sparse
- Ensure observation space includes relevant info
- Increase exploration (`ent_coef` for PPO)
- Try different random seed

**Problem: Agent gets stuck in local optimum**
- Increase training steps
- Adjust reward shaping
- Randomize initial conditions more

**Problem: Trained model doesn't work in test**
- Make sure to load VecNormalize stats
- Check if deterministic=True in test
- Verify environment matches training setup

## 🎓 Learning Resources

- [Stable-Baselines3 Documentation](https://stable-baselines3.readthedocs.io/)
- [Gymnasium Documentation](https://gymnasium.farama.org/)
- [MuJoCo Documentation](https://mujoco.readthedocs.io/)
- [PPO Paper](https://arxiv.org/abs/1707.06347)
- [SAC Paper](https://arxiv.org/abs/1801.01290)

## 💡 Tips for Success

1. **Start Simple**: Use PPO with default settings
2. **Monitor Training**: Always use TensorBoard
3. **Be Patient**: RL training takes time (100k+ steps)
4. **Shape Rewards**: Dense rewards learn faster than sparse
5. **Normalize Inputs**: VecNormalize is crucial for stability
6. **Test Often**: Evaluate every 10k steps to catch issues early
7. **Save Checkpoints**: Don't lose progress if training crashes

## 🚀 Next Steps

Once you have a working agent:

1. **Improve Performance**: Tune hyperparameters, adjust rewards
2. **Harder Tasks**: Pick and place at target location
3. **Multiple Objects**: Learn to pick different objects
4. **Real Robot**: Transfer learned policy to physical robot
5. **Vision-Based**: Use images instead of position observations

## 📝 License

This project is for educational purposes.
