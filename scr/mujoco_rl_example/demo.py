"""
Demo script to visualize the robot arm environment before training.
This lets you see the task and test random actions.
"""
import time
import numpy as np
from robot_arm_env import RobotArmPickEnv


def demo_environment(n_episodes=3, steps_per_episode=100):
    """
    Run a demo of the environment with random actions.
    
    Args:
        n_episodes: Number of episodes to run
        steps_per_episode: Steps per episode
    """
    print("🤖 Robot Arm Environment Demo")
    print("="*60)
    print("\nThis demo shows the robot arm attempting to pick up the cube")
    print("using RANDOM actions (no learning yet).")
    print("\nWatch how:")
    print("  - The robot moves its joints randomly")
    print("  - The cube responds to physics")
    print("  - The gripper tries to grasp (but fails without learning)")
    print("\nAfter training with RL, the robot will learn to:")
    print("  1. Reach toward the cube")
    print("  2. Grasp it with the gripper")
    print("  3. Lift it up successfully")
    print("="*60)
    print()
    
    # Create environment with rendering
    env = RobotArmPickEnv(render_mode="human")
    
    print(f"🎬 Running {n_episodes} episodes with random actions...")
    print()
    
    for episode in range(n_episodes):
        print(f"Episode {episode + 1}/{n_episodes}")
        
        # Reset environment
        obs, info = env.reset()
        
        total_reward = 0
        best_distance = float('inf')
        
        for step in range(steps_per_episode):
            # Take random action
            action = env.action_space.sample()
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            
            total_reward += reward
            best_distance = min(best_distance, info['distance_to_cube'])
            
            # Render
            env.render()
            time.sleep(0.02)  # Slow down for visualization
            
            # Check if done
            if terminated or truncated:
                break
        
        # Print episode summary
        print(f"  Reward: {total_reward:.2f}")
        print(f"  Best distance to cube: {best_distance:.3f}m")
        print(f"  Final cube height: {info['cube_height']:.3f}m")
        print(f"  Success: {'✓' if info['is_success'] else '✗'}")
        print()
    
    print("="*60)
    print("Demo complete!")
    print()
    print("As you can see, random actions don't solve the task.")
    print("This is why we need Reinforcement Learning! 🚀")
    print()
    print("Next steps:")
    print("  1. Run: python train.py")
    print("  2. Wait for training (~30-60 min)")
    print("  3. Run: python test.py")
    print("  4. Watch the trained robot succeed!")
    print("="*60)
    
    env.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Demo robot arm environment")
    parser.add_argument(
        "--episodes",
        type=int,
        default=3,
        help="Number of episodes to run (default: 3)"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=100,
        help="Steps per episode (default: 100)"
    )
    
    args = parser.parse_args()
    
    demo_environment(n_episodes=args.episodes, steps_per_episode=args.steps)
