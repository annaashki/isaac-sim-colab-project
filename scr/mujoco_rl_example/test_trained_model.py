"""
Test and visualize the trained RL model picking up the cube.
"""
import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from robot_arm_env import RobotArmPickEnv


def test_model(model_path, n_episodes=5, render=True):
    """
    Load and test a trained model.
    
    Args:
        model_path: Path to the saved model directory
        n_episodes: Number of episodes to test
        render: Whether to render the environment
    """
    print(f"Loading model from: {model_path}")
    
    # Create environment
    render_mode = "human" if render else None
    env = RobotArmPickEnv(render_mode=render_mode)
    env = DummyVecEnv([lambda: env])
    
    # Load normalization statistics
    vec_normalize_path = os.path.join(model_path, "vec_normalize.pkl")
    if os.path.exists(vec_normalize_path):
        print("Loading VecNormalize statistics...")
        env = VecNormalize.load(vec_normalize_path, env)
        env.training = False  # Don't update stats during testing
        env.norm_reward = False
    
    # Load model
    model = PPO.load(os.path.join(model_path, "best_model.zip"), env=env)
    print("Model loaded successfully!\n")
    
    # Test the model
    successes = 0
    total_rewards = []
    
    for episode in range(n_episodes):
        obs = env.reset()
        episode_reward = 0
        done = False
        step = 0
        
        print(f"\n{'='*60}")
        print(f"Episode {episode + 1}/{n_episodes}")
        print(f"{'='*60}")
        
        while not done:
            # Predict action using the trained model
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            
            episode_reward += reward[0]
            step += 1
            
            # Print info every 10 steps
            if step % 10 == 0:
                cube_height = info[0].get('cube_height', 0)
                distance = info[0].get('distance_to_cube', 0)
                print(f"  Step {step:3d} | Reward: {reward[0]:6.2f} | "
                      f"Cube height: {cube_height:.3f} | Distance: {distance:.3f}")
            
            if done:
                success = info[0].get('is_success', False)
                if success:
                    successes += 1
                    print(f"\n✅ SUCCESS! Cube picked up!")
                else:
                    print(f"\n❌ Episode ended without success.")
                
                total_rewards.append(episode_reward)
                print(f"Episode reward: {episode_reward:.2f}")
                break
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Test Summary")
    print(f"{'='*60}")
    print(f"Episodes: {n_episodes}")
    print(f"Successes: {successes}/{n_episodes} ({100*successes/n_episodes:.1f}%)")
    print(f"Average reward: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")
    print(f"{'='*60}\n")
    
    env.close()
    
    return successes / n_episodes


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test trained RL model")
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to model directory (e.g., ./models/PPO_20231215_123456/best_model)"
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
        help="Disable rendering"
    )
    
    args = parser.parse_args()
    
    # If no model path provided, find the most recent best_model
    if args.model_path is None:
        models_dir = "./models"
        if os.path.exists(models_dir):
            # Get all subdirectories
            subdirs = [os.path.join(models_dir, d) for d in os.listdir(models_dir)
                      if os.path.isdir(os.path.join(models_dir, d))]
            if subdirs:
                # Sort by modification time, get most recent
                latest_dir = max(subdirs, key=os.path.getmtime)
                args.model_path = os.path.join(latest_dir, "best_model")
                print(f"Auto-detected model: {args.model_path}")
            else:
                print("Error: No trained models found in ./models/")
                exit(1)
        else:
            print("Error: ./models/ directory not found")
            exit(1)
    
    if not os.path.exists(args.model_path):
        print(f"Error: Model not found at {args.model_path}")
        exit(1)
    
    # Test the model
    success_rate = test_model(
        args.model_path,
        n_episodes=args.episodes,
        render=not args.no_render
    )
