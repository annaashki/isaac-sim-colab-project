"""
Test/inference script for trained robot arm RL model.
Loads a trained model and visualizes it picking up the cube.
"""
import os
import time
import argparse
import numpy as np
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from robot_arm_env import RobotArmPickEnv


def test_trained_model(
    model_path,
    vec_normalize_path=None,
    algorithm="PPO",
    n_episodes=5,
    render=True
):
    """
    Test a trained RL model.
    
    Args:
        model_path: Path to the saved model (.zip file)
        vec_normalize_path: Path to VecNormalize stats (usually vec_normalize.pkl)
        algorithm: Algorithm used ("PPO" or "SAC")
        n_episodes: Number of episodes to run
        render: Whether to render the environment
    """
    print(f"Loading trained {algorithm} model from: {model_path}")
    
    # Create environment
    render_mode = "human" if render else None
    env = RobotArmPickEnv(render_mode=render_mode)
    
    # Wrap in DummyVecEnv to match training setup
    env = DummyVecEnv([lambda: env])
    
    # Load VecNormalize if available
    if vec_normalize_path and os.path.exists(vec_normalize_path):
        print(f"Loading VecNormalize stats from: {vec_normalize_path}")
        env = VecNormalize.load(vec_normalize_path, env)
        env.training = False  # Don't update stats during testing
        env.norm_reward = False  # Don't normalize rewards during testing
    
    # Load the trained model
    if algorithm == "PPO":
        model = PPO.load(model_path, env=env)
    elif algorithm == "SAC":
        model = SAC.load(model_path, env=env)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    
    print(f"\nRunning {n_episodes} test episodes...")
    print("="*60)
    
    # Statistics
    episode_rewards = []
    episode_successes = []
    episode_lengths = []
    
    for episode in range(n_episodes):
        obs = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        success = False
        
        while not done:
            # Get action from model (deterministic for testing)
            action, _states = model.predict(obs, deterministic=True)
            
            # Take step
            obs, reward, done, info = env.step(action)
            
            episode_reward += reward[0]
            episode_length += 1
            
            # Check success
            if info[0].get('is_success', False):
                success = True
            
            # Render if enabled
            if render:
                time.sleep(0.01)  # Slow down for visualization
        
        # Store statistics
        episode_rewards.append(episode_reward)
        episode_successes.append(success)
        episode_lengths.append(episode_length)
        
        # Print episode results
        print(f"Episode {episode + 1}/{n_episodes}:")
        print(f"  Reward: {episode_reward:.2f}")
        print(f"  Length: {episode_length} steps")
        print(f"  Success: {'✓' if success else '✗'}")
        print()
    
    # Print summary statistics
    print("="*60)
    print("Summary Statistics:")
    print(f"  Average Reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"  Average Length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f} steps")
    print(f"  Success Rate: {np.mean(episode_successes) * 100:.1f}% ({sum(episode_successes)}/{n_episodes})")
    print("="*60)
    
    env.close()
    
    return {
        'rewards': episode_rewards,
        'successes': episode_successes,
        'lengths': episode_lengths,
        'success_rate': np.mean(episode_successes)
    }


def find_best_model(models_dir):
    """Find the best model in the models directory."""
    # Look for best_model directory
    for root, dirs, files in os.walk(models_dir):
        if "best_model" in root and "best_model.zip" in files:
            return os.path.join(root, "best_model.zip")
    
    # Look for final_model directory
    for root, dirs, files in os.walk(models_dir):
        if "final_model" in root:
            for file in files:
                if file.endswith(".zip"):
                    return os.path.join(root, file)
    
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test trained RL model for robot arm")
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to the trained model (.zip file). If not provided, will search for best model."
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default="./models",
        help="Directory containing trained models (used if --model-path not specified)"
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        default="PPO",
        choices=["PPO", "SAC"],
        help="RL algorithm used (default: PPO)"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=5,
        help="Number of test episodes (default: 5)"
    )
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="Disable rendering (run headless)"
    )
    
    args = parser.parse_args()
    
    # Find model path
    if args.model_path is None:
        print(f"Searching for best model in {args.models_dir}...")
        model_path = find_best_model(args.models_dir)
        if model_path is None:
            print(f"Error: No trained model found in {args.models_dir}")
            print("Please train a model first using train.py")
            exit(1)
        print(f"Found model: {model_path}")
    else:
        model_path = args.model_path
        if not os.path.exists(model_path):
            print(f"Error: Model not found at {model_path}")
            exit(1)
    
    # Look for VecNormalize stats in the same directory
    model_dir = os.path.dirname(model_path)
    vec_normalize_path = os.path.join(model_dir, "vec_normalize.pkl")
    if not os.path.exists(vec_normalize_path):
        vec_normalize_path = None
    
    # Test the model
    results = test_trained_model(
        model_path=model_path,
        vec_normalize_path=vec_normalize_path,
        algorithm=args.algorithm,
        n_episodes=args.episodes,
        render=not args.no_render
    )
    
    print(f"\n✓ Testing complete!")
