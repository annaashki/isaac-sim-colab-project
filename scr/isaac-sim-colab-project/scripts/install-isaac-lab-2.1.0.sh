#! /bin/bash -e

# This script is intended to be used with a specially modified Colab notebook.
# i.e., https://colab.research.google.com/github/j3soon/isaac-sim-colab/blob/main/notebooks/isaac-lab-2.1.0-colab.ipynb
# For more information, see: https://github.com/j3soon/isaac-sim-colab.

# Confirm that Isaac Sim is installed
which isaacsim

# Ref: https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip_installation.html
apt-get install -y cmake build-essential
uv pip install -qq torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu118
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab
git checkout v2.1.0
./isaaclab.sh --install

# Set environment variables
# Ref: https://docs.isaacsim.omniverse.nvidia.com/latest/installation/install_python.html#running-isaac-sim
export OMNI_KIT_ACCEPT_EULA=YES

# Run minimal example
# Ref: https://isaac-sim.github.io/IsaacLab/main/source/overview/reinforcement-learning/rl_existing_scripts.html
# python scripts/reinforcement_learning/rl_games/train.py --task Isaac-Cartpole-v0 --headless
