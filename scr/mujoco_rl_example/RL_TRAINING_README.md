# Robot Arm Pick-and-Place with Reinforcement Learning

## Overview
This project uses **Proximal Policy Optimization (PPO)** to train a 6-axis robot arm to pick up a cube in a MuJoCo simulation.

## Training Progress

### Current Status
- **Algorithm**: PPO (Proximal Policy Optimization)
- **Training Timesteps**: 100,000
- **Current Progress**: ~12,288 timesteps (12% complete)
- **Performance**: Reward improving from 2.6 → 5.25

### Training Details
- **Observation Space**: 21 dimensions
  - 6 joint positions
  - 6 joint velocities
  - 3 gripper position (x, y, z)
  - 3 cube position (x, y, z)
  - 3 relative position (cube - gripper)
tail -f training_improved.log
- **Action Space**: 6 continuous actions (normalized to [-1, 1])
  - Base rotation
  - 3 elbow joints
  - Gripper rotation
  - Gripper fingers

- **Reward Function**:
  1. **Approach reward**: Exponential reward for getting close to cube
  2. **Contact bonus**: Extra reward when gripper is very close (<0.15m)
  3. **Lift reward**: Reward for lifting the cube up
  4. **Success bonus**: Large reward (50.0) for lifting cube above 0.7m

## Files

### Training
- `robot_arm_env.py` - Custom Gymnasium environment with robot arm and cube
- `train.py` - PPO training script with checkpoints and evaluation
- `training_output.log` - Live training log

### Testing
- `test_trained_model.py` - Script to test and visualize trained model
- `demo.py` - Interactive demonstration
- `interactive_control.py` - Manual control of robot arm

## Usage

### Monitor Training Progress
```bash
# Check training log
tail -f training_output.log

# View with TensorBoard (in another terminal)
tensorboard --logdir ./logs
```

### Test Trained Model
Once training completes, test the model:
```bash
python test_trained_model.py --episodes 5
```

Or specify a specific model:
```bash
python test_trained_model.py --model-path ./models/PPO_20251211_103208/best_model --episodes 10
```

### Training Output Location
- **Models**: `./models/PPO_YYYYMMDD_HHMMSS/`
  - `checkpoints/` - Saved every 50k steps
  - `best_model/` - Best performing model during evaluation
  - `eval_logs/` - Evaluation metrics
- **Logs**: `./logs/PPO_YYYYMMDD_HHMMSS/` - TensorBoard logs

## Expected Results

As training progresses, you should see:
1. **Early training (0-20k steps)**: Robot explores randomly, low success rate
2. **Mid training (20k-60k steps)**: Robot learns to approach cube, some successful grasps
3. **Late training (60k-100k steps)**: Consistent picking behavior, high success rate (>70%)

## Training Parameters

```python
PPO(
    learning_rate=3e-4,
    n_steps=2048,        # Steps per update
    batch_size=64,
    n_epochs=10,         # Optimization epochs per update
    gamma=0.99,          # Discount factor
    gae_lambda=0.95,     # GAE parameter
    clip_range=0.2,      # PPO clip range
    ent_coef=0.01,       # Entropy coefficient for exploration
)
```

## Success Criteria

The robot successfully picks up the cube when:
- Cube height > 0.7m (lifted from initial position of 0.5m)
- Gripper distance to cube < 0.15m

## Tips

- **Longer training**: For better results, train for 500k-1M timesteps
- **Hyperparameters**: Adjust learning_rate, ent_coef for different exploration/exploitation balance
- **Reward shaping**: Modify `_compute_reward()` in robot_arm_env.py to change learning behavior
- **Visualization**: Use `train_with_rendering.py` to see training in real-time (slower but insightful)
