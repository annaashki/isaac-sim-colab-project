#!/bin/bash

# Quick start script for robot arm RL training

echo "🤖 Robot Arm RL Training - Quick Start"
echo "======================================"
echo ""

# Check if in correct directory
if [ ! -f "robot_arm_env.py" ]; then
    echo "❌ Error: Please run this script from the mujoco_rl_example directory"
    exit 1
fi

# Install dependencies
echo "📦 Step 1: Installing dependencies..."
echo "This may take a few minutes..."
pip install gymnasium stable-baselines3[extra] tensorboard

echo ""
echo "✅ Dependencies installed!"
echo ""

# Quick test
echo "🧪 Step 2: Testing environment..."
python -c "from robot_arm_env import RobotArmPickEnv; env = RobotArmPickEnv(); print('Environment created successfully!'); env.close()"

if [ $? -eq 0 ]; then
    echo "✅ Environment test passed!"
else
    echo "❌ Environment test failed. Check error messages above."
    exit 1
fi

echo ""
echo "======================================"
echo "🎓 Setup Complete!"
echo "======================================"
echo ""
echo "Next steps:"
echo ""
echo "1️⃣  Start training:"
echo "    python train.py --algorithm PPO --timesteps 500000"
echo ""
echo "2️⃣  Monitor training (in a new terminal):"
echo "    tensorboard --logdir ./logs"
echo ""
echo "3️⃣  Test trained model:"
echo "    python test.py --episodes 5"
echo ""
echo "💡 Tip: Training will take 30-60 minutes. Be patient!"
echo ""
