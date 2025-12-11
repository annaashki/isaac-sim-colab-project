#!/bin/bash

# Improved training script with better hyperparameters
# Run this to retrain the robot with the improved environment

echo "=========================================="
echo "Starting Improved RL Training"
echo "=========================================="
echo ""
echo "Key Improvements:"
echo "  - Faster actions (50 steps vs 1000)"
echo "  - Longer episodes (500 steps vs 100)"
echo "  - Better reward shaping"
echo "  - Reduced PID gains (less oscillation)"
echo ""
echo "Training for 500,000 timesteps (~1 hour)"
echo "=========================================="
echo ""

cd "$HOME/Library/CloudStorage/OneDrive-AZCollaboration/PythonProjects/WinterSchool/scr/mujoco_rl_example"

nohup "$HOME/Library/CloudStorage/OneDrive-AZCollaboration/PythonProjects/WinterSchool/.venv/bin/python" train.py \
    --algorithm PPO \
    --timesteps 500000 \
    --save-dir ./models \
    --log-dir ./logs \
    > training_improved.log 2>&1 &

TRAIN_PID=$!
echo "Training started with PID: $TRAIN_PID"
echo ""
echo "Monitor progress with:"
echo "  tail -f training_improved.log"
echo ""
echo "Check tensorboard:"
echo "  tensorboard --logdir ./logs"
echo ""
